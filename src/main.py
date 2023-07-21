import sys
sys.path.append("../")
sys.path.insert(0, '../KE-GCN/')
import copy
# from torch.utils.data import random_split
# from torch.utils.tensorboard import SummaryWriter
# from dgl.contrib.data import load_data
import numpy as np
# from RGCN.model import Model
from models import AutoRGCN_Align
from metrics import *
from utils import *
from server import Server
from client import Client
import argparse
import json
import logging
import os
import random
from sklearn.cluster import SpectralClustering
import pickle

# init directories
def init_dir(args):
    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)
    if not os.path.exists(args.state_dir + args.run_mode + '/'):
        os.makedirs(args.state_dir + args.run_mode + '/')

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)


# init logger
def init_logger(args):
    log_file = os.path.join(args.log_dir, args.run_mode + '.log')

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# init FL setting
def init_fed(args):
    # load data
    data = load_data(dataset=args.dataset)
    # keep the same number of node and relation to each client 
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes

    # create sub graph for each client
    # A. Extract the subgraph
    # client_edge_num = len(data.edge_type) // args.num_client
    # edge_num_list = [client_edge_num] * args.num_client
    # if client_edge_num * args.num_client != len(data.edge_type):
    #     edge_num_list[0] = client_edge_num + len(data.edge_type) - client_edge_num * args.num_client
    # edge_perms = random_split(range(len(data.edge_type)), edge_num_list)

    # B. Extract the subgraph
    # edge_perms = []
    # for _ in range(args.num_client):
    #     perm = np.random.permutation(len(data.edge_type))[:int(len(data.edge_type)*args.labeled_rate)]
    #     edge_perms.append(perm)

    # C. Create non-iid data
    client_edge_num_1 = int(len(data.edge_type)*2/4)
    client_edge_num_2 = int(len(data.edge_type)*1/4)
    client_edge_num_other = int(len(data.edge_type)*1/32)
    edge_num_list = [client_edge_num_1, client_edge_num_2] + [client_edge_num_other]*8
    if client_edge_num_1 + client_edge_num_2 + 8*client_edge_num_other != len(data.edge_type):
        edge_num_list[0] = client_edge_num_1 + len(data.edge_type) - (client_edge_num_1 + client_edge_num_2+ 8*client_edge_num_other)
    edge_perms = random_split(range(len(data.edge_type)), edge_num_list)

    datasets = []
    for perm in edge_perms:
        client_data = copy.deepcopy(data)

        train_idx = client_data.train_idx
        np.random.shuffle(train_idx)
        client_data.train_idx = train_idx[:int(len(train_idx)*args.labeled_rate)]

        client_data.edge_src = client_data.edge_src[perm]
        client_data.edge_dst = client_data.edge_dst[perm]
        client_data.edge_type = client_data.edge_type[perm]
        client_data.edge_norm = client_data.edge_norm[perm]

        norm_matrix = np.zeros((num_nodes, num_rels))
        for i in range(len(perm)):
            norm_matrix[client_data.edge_dst[i]][client_data.edge_type[i]] += 1
        
        for i in range(len(perm)):
            client_data.edge_norm[i] = 1 / norm_matrix[client_data.edge_dst[i]][client_data.edge_type[i]]

        datasets.append(client_data)

    # init the roles of FL
    # 1.init server
    ser_model = Model(num_nodes, args.n_hidden, num_classes, num_rels, args.n_bases, args.n_hidden_layers, args.gpu).to(args.gpu)
    server = Server(ser_model, args.gpu, logging, args.writer)
    # 2.init clients
    clients = []
    for i in range(args.num_client):
        model = Model(num_nodes, args.n_hidden, num_classes, num_rels, args.n_bases, args.n_hidden_layers, args.gpu).to(args.gpu)
        client = Client(i, datasets[i], model, args.gpu, args.local_epoch, args.lr, args.l2norm, args.state_dir + args.run_mode + "/", logging, args.writer)
        clients.append(client)
    
    return server, clients

def split_data_evenly(data, number_of_clients):
    random.shuffle(data)
    client_row_num = len(data)  // number_of_clients
    row_num_list = [client_row_num] * number_of_clients
    if client_row_num * number_of_clients != len(data):
        row_num_list[0] = client_row_num + len(data) - client_row_num * number_of_clients
    client_data = []
    row_start = 0
    for row_num in row_num_list:
        client_data.append(data[row_start:row_start+row_num])
        row_start += row_num
    return client_data

def basis_paper_split(KG, train_idx):
    n_types = {0: 146, 1:5, 2:28, 3:237, 4: 78, 5:1318, 6: 5450}
    # for node, node_type, true in zip(cx.row, cx.col, cx.data):
    #     n_types[node_type] += 1

    N = 10
    train_shuffled = train_idx.copy()
    random.shuffle(train_shuffled)
    # print(train_idx)
    # print(train_shuffled)
    new_train = np.array_split(train_shuffled, N)

    all_nodes = 8285

    not_included = random.randint(0, 6)
    client_base = []
    for i in range(N):
        running_total = 0
        for name, length in n_types.items():
            if name != not_included:
                running_total += random.randint(0, length)
        
        node_set = set()

        total_nodes = [i for i in range(all_nodes)]
        node_set.update(random.sample(total_nodes, running_total))

        node_set.update(new_train[i])
        client_base.append(node_set)
    

    return client_base, new_train


def get_extended_adj_client(e, KG, client_base_nodes):
    nei_list = []
    ent_row, rel_row = [], []
    ent_col, rel_col = [], []
    ent_data, rel_data = [], []
    count = 0
    for tri in KG:
        if tri[0] in client_base_nodes and tri[2] in client_base_nodes:
            nei_list.append([tri[0], tri[1], tri[2]])
            ent_row.append(tri[0])
            ent_col.append(count)
            ent_data.append(1.)
            ent_row.append(tri[2])
            ent_col.append(count)
            ent_data.append(1.)
            rel_row.append(tri[1])
            rel_col.append(count)
            rel_data.append(1.)
        count += 1
    ent_adj_ind = sp.coo_matrix((ent_data, (ent_row, ent_col)), shape=(e, count))
    rel_adj_ind = sp.coo_matrix((rel_data, (rel_row, rel_col)), shape=(max(rel_row)+1, count))
    return [ent_adj_ind, rel_adj_ind, np.array(nei_list)]
    




        




# def split_data_weighted_to_client_1(data, number_of_clients):
#     # clustering = SpectralClustering(n_clusters=number_of_clients, affinity = "precomputed").fit_predict(data)
#     clustering = SpectralClustering(n_clusters=number_of_clients, affinity = "precomputed_nearest_neighbors").fit_predict(data)
#     print(clustering)
#     print(clustering.shape)
#     print(data.shape)
#     total = {0: 0, 1:0, 2:0, 3:0, 4:0}
#     for val in clustering:
#         total[val] += 1
#     print(total)
    # return clustering

    # random.shuffle(data)
    # client_row_num = len(data)  // number_of_clients
    # row_num_list = [client_row_num] * number_of_clients
    # if client_row_num * number_of_clients != len(data):
    #     row_num_list[0] = client_row_num + len(data) - client_row_num * number_of_clients
    # client_data = []
    # row_start = 0
    # for row_num in row_num_list:
    #     client_data.append(data[row_start:row_start+row_num])
    #     row_start += row_num
    # return client_data

def split_graph(adj):

    client_edge_num_1 = int(len(adj)*2/4)
    client_edge_num_2 = int(len(adj)*1/4)
    client_edge_num_other = int(len(adj)*1/32)
    edge_num_list = [client_edge_num_1, client_edge_num_2] + [client_edge_num_other]*8
    if client_edge_num_1 + client_edge_num_2 + 8*client_edge_num_other != len(adj):
        edge_num_list[0] = client_edge_num_1 + len(adj) - (client_edge_num_1 + client_edge_num_2+ 8*client_edge_num_other)
    client_adj = []
    row_start = 0
    for row_num in edge_num_list:
        client_adj.append(adj[row_start:row_start+row_num])
        row_start += row_num
    return client_adj



def classification(args):
    '''
    code for entity classification task
    '''
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    # flags.DEFINE_string('dataset', 'aifb', 'Dataset: am, wordnet, aifb.')
    flags.DEFINE_string('dataset', 'aifb', 'Dataset: am, wordnet, aifb.')
    # flags.DEFINE_string('mode', 'TransE', 'KE method for GCN: TransE, TransH, TransD, DistMult, RotatE, QuatE')
    flags.DEFINE_string('mode', 'None', 'KE method for GCN: TransE, TransH, TransD, DistMult, RotatE, QuatE')
    # flags.DEFINE_string('optim', 'Adam', 'Optimizer: GD, Adam')
    flags.DEFINE_string('optim', 'Adam', 'Optimizer: GD, Adam')
    flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
    flags.DEFINE_integer('num_negs', 5, 'Number of negative samples for each positive seed.')
    flags.DEFINE_float('alpha', 0.5, 'Weight of entity conv update.')
    flags.DEFINE_float('beta', 0.5, 'Weight of relation conv update.')
    flags.DEFINE_integer('layer', 0, 'number of hidden layers')
    flags.DEFINE_integer('dim', 32, 'hidden Dimension')
    flags.DEFINE_integer('randomseed', 12306, 'seed for randomness')
    flags.DEFINE_boolean('rel_update', False, 'If true, use graph conv for rel update.')
    flags.DEFINE_boolean('valid', False, 'If true, split validation data.')
    flags.DEFINE_boolean('save', False, 'If true, save the print')
    flags.DEFINE_string('metric', "cityblock", 'metric for testing')
    flags.DEFINE_string('loss_mode', "L1", 'mode for loss calculation')
    flags.DEFINE_string('embed', "random", 'init embedding for entities')
    flags.DEFINE_string('run_mode', "Fed", 'initial run mode')


    # CREATE MODELS FOR GRID SEARCH
    # with open('combos.pkl', 'rb') as file:
    #     combos = pickle.load(file)

    # my_task_id = int(sys.argv[1])
    # num_tasks = int(sys.argv[2])

    # # Assign indices to this process/task
    # my_combos = combos[my_task_id:len(combos):num_tasks]

    my_combos = [(100, 0.01)]
    for basis, rate in my_combos:
        np.random.seed(FLAGS.randomseed)
        random.seed(FLAGS.randomseed)
        tf.set_random_seed(FLAGS.randomseed)

        flags.DEFINE_float('learning_rate', rate, 'Initial learning rate.')
        flags.DEFINE_float('basis', basis, 'Initial learning rate.')

        if FLAGS.save:
            nsave = "../log/{}/{}".format(FLAGS.dataset, FLAGS.mode)
        else:
            print("not saving file")
            nsave = "log/trash"
        create_exp_dir(nsave)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode="w")
        save_fname = 'alpha{}-beta{}-layer{}-sdim{}-lr{}-seed{}'.format(
                    FLAGS.alpha, FLAGS.beta, FLAGS.layer, FLAGS.dim,
                    rate, FLAGS.randomseed)

        save_fname = "auto-" + save_fname
        if not FLAGS.valid:
            save_fname = "test-" + save_fname
        fh = logging.FileHandler(os.path.join(nsave, save_fname + ".txt"), "w")
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.getLogger().setLevel(logging.INFO)

        # Load data
        adj, num_ent, og_train, test, valid, y, KG = load_data_class(FLAGS)
        # real_adj = adj[-1]
        client_data = {}
        # data = {"train": train, "test": test, "valid": valid, "y": y}

        # number_of_clients = 5

        # HERE FOR SPLITTING DATA
        # train = split_data_evenly(train, args.num_client)
        # test = split_data_evenly(test, args.num_client)
        # valid = split_data_evenly(valid, args.num_client)


        new_client_base, new_train_split = basis_paper_split(KG, og_train)

        client_nodes_num = [len(client_base) for client_base in new_client_base]

        for client_index, client_base_nodes in enumerate(new_client_base):
            client_adj = get_extended_adj_client(num_ent, KG, client_base_nodes)
            support = [preprocess_adj(client_adj)]

            # support = [preprocess_adj(adj)]

            client_data[client_index] = {"support": support, "train": new_train_split[client_index], "test": test, "valid": valid, "y": y}
            # client_data[client_index] = {"support": support, "train": new_train_split[client_index], "test": test, "valid": valid, "y": y}

        # train = split_data_weighted_to_client_1(real_adj, number_of_clients)
        # test = split_data_weighted_to_client_1(test, args.num_client)
        # valid = split_data_weighted_to_client_1(valid, args.num_client)

        # for client_index in range(args.num_client):
        #     client_data[client_index] = {"train": train[client_index], "test": test[client_index], "valid": valid[client_index], "y": y}

        og_train = [og_train, y]
        train = [[t, y] for t in new_train_split]
        # train = [[t, y] for t in new_train_split]
        rel_num = np.max(adj[2][:, 1]) + 1
        # print("Relation num: ", rel_num)

        # process graph to fit into later computation
        support = [preprocess_adj(adj)]
        # for key, data_value in client_data.items():
        #     data_value["support"] = support
        num_supports = 1
        model_func = AutoRGCN_Align
        num_negs = FLAGS.num_negs
        class_num = y.shape[1]
        # print("Entity num: ", num_ent)
        # print("Class num: ", class_num)

        if FLAGS.dataset == "fb15k":
            task = "label"
            get_eval = get_label
        else:
            task = "class"
            get_eval = get_class

        # Define placeholders
        placeholders = {
            'features': tf.placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder_with_default(0, shape=())
        }
        placeholders['support'] = [[tf.placeholder(tf.float32, shape=[None, 1]),
                            tf.placeholder(tf.float32, shape=[None, 1]), \
                            tf.placeholder(tf.int32)] for _ in range(num_supports)]

        # Create model
        input_dim = [num_ent, rel_num]
        hidden_dim = [FLAGS.dim, FLAGS.dim]
        output_dim = [class_num, FLAGS.dim]
        if FLAGS.mode == "TransH":
            hidden_dim[1] *= 2
        elif FLAGS.mode == "TransD":
            hidden_dim[0] *= 2
            hidden_dim[1] *= 2

        # init the roles of FL
        # 1.init server

        ## basis decomp False and num_bases = -1 for non basis decomp
        ser_model =  model_func(placeholders, input_dim, hidden_dim, output_dim, dataset=FLAGS.dataset,
                            train_labels=train[0], mode=FLAGS.mode, embed=FLAGS.embed, alpha=FLAGS.alpha,
                            beta=FLAGS.beta, layer_num=FLAGS.layer, basis_decomp = True, num_bases = basis, sparse_inputs=False, featureless=True,
                            logging=True, client_id = -1, rel_update=FLAGS.rel_update, task=task)

        # server = Server(ser_model, args.gpu, logging, client_data[0], get_eval, writer = None)
        server = Server(ser_model, 0, logging, client_data[0], get_eval, writer = None)

        # init basis dict to random
        init_basis_dict = {}
        for i in range(10):
            init_basis_dict[i] = {0:np.array([0,0])}

        # 2.init clients
        clients = []
        for i in range(10):
            model =  model_func(placeholders, input_dim, hidden_dim, output_dim, dataset=FLAGS.dataset,
                            train_labels=train[i], mode=FLAGS.mode, embed=FLAGS.embed, alpha=FLAGS.alpha,
                            beta=FLAGS.beta, layer_num=FLAGS.layer, sparse_inputs=False, featureless=True,
                            logging=True, logger = logging, basis_decomp = True, num_bases = basis, num_nodes=client_nodes_num[i],
                            client_id=i, basis_dict=init_basis_dict, rel_update=FLAGS.rel_update, task=task)

            client = Client(i, client_data[i], model, FLAGS, FLAGS.epochs, FLAGS.learning_rate, 0, '../log/state/' + "Single" + "/", logging, get_eval)
            clients.append(client)
        
        return server, clients, FLAGS


# FL process
def FedRunning(args, server, clients, FLAGS):

    client_basis_dict = {}
    for i, client in enumerate(clients):
        new_basis = client.getBasis()
        client_basis_dict[i] = new_basis

    for client in np.array(clients):
        client.setAllBasis(client_basis_dict)

    for t in range(20):
    flags.DEFINE_string('run_mode', "Single", 'initial run mode')

    np.random.seed(FLAGS.randomseed)
    random.seed(FLAGS.randomseed)
    tf.set_random_seed(FLAGS.randomseed)

    if FLAGS.save:
        nsave = "../log/{}/{}".format(FLAGS.dataset, FLAGS.mode)
    else:
        print("not saving file")
        nsave = "log/trash"
    create_exp_dir(nsave)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode="w")
    save_fname = 'alpha{}-beta{}-layer{}-sdim{}-lr{}-seed{}'.format(
                FLAGS.alpha, FLAGS.beta, FLAGS.layer, FLAGS.dim,
                FLAGS.learning_rate, FLAGS.randomseed)

    save_fname = "auto-" + save_fname
    if not FLAGS.valid:
        save_fname = "test-" + save_fname
    fh = logging.FileHandler(os.path.join(nsave, save_fname + ".txt"), "w")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    # Load data
    adj, num_ent, train, test, valid, y = load_data_class(FLAGS)
    data = {"train": train, "test": test, "valid": valid, "y": y}


    train = [train, y]
    rel_num = np.max(adj[2][:, 1]) + 1
    print("Relation num: ", rel_num)

    # process graph to fit into later computation
    support = [preprocess_adj(adj)]
    data["support"] = support
    num_supports = 1
    model_func = AutoRGCN_Align
    num_negs = FLAGS.num_negs
    class_num = y.shape[1]
    print("Entity num: ", num_ent)
    print("Class num: ", class_num)

    if FLAGS.dataset == "fb15k":
        task = "label"
        get_eval = get_label
    else:
        task = "class"
        get_eval = get_class

    # Define placeholders
    placeholders = {
        'features': tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder_with_default(0, shape=())
    }
    placeholders['support'] = [[tf.placeholder(tf.float32, shape=[None, 1]),
                        tf.placeholder(tf.float32, shape=[None, 1]), \
                        tf.placeholder(tf.int32)] for _ in range(num_supports)]

    # Create model
    input_dim = [num_ent, rel_num]
    hidden_dim = [FLAGS.dim, FLAGS.dim]
    output_dim = [class_num, FLAGS.dim]
    if FLAGS.mode == "TransH":
        hidden_dim[1] *= 2
    elif FLAGS.mode == "TransD":
        hidden_dim[0] *= 2
        hidden_dim[1] *= 2

    # init the roles of FL
    # 1.init server
    ser_model =  model_func(placeholders, input_dim, hidden_dim, output_dim, dataset=FLAGS.dataset,
                        train_labels=train, mode=FLAGS.mode, embed=FLAGS.embed, alpha=FLAGS.alpha,
                        beta=FLAGS.beta, layer_num=FLAGS.layer, sparse_inputs=False, featureless=True,
                        logging=True, rel_update=FLAGS.rel_update, task=task)

    server = Server(ser_model, args.gpu, logging)
    # 2.init clients
    clients = []
    for i in range(args.num_client):
        model =  model_func(placeholders, input_dim, hidden_dim, output_dim, dataset=FLAGS.dataset,
                        train_labels=train, mode=FLAGS.mode, embed=FLAGS.embed, alpha=FLAGS.alpha,
                        beta=FLAGS.beta, layer_num=FLAGS.layer, sparse_inputs=False, featureless=True,
                        logging=True, rel_update=FLAGS.rel_update, task=task)

        client = Client(i, data, model, FLAGS, args.local_epoch, args.lr, args.l2norm, args.state_dir + "Single" + "/", logging, get_eval)
        clients.append(client)
    
    return server, clients


# FL process
def FedRunning(args, server, clients):

    for t in range(args.round):

        logging.info(f"---------------------Round {t}---------------------")

        # The 0 step
        # perm = list(range(10))
        # random.shuffle(perm)
        # perm = np.array(perm[:int(10 * 1)])


        # The 1 step
        for i, client in enumerate(clients):
            if t != 0:
                client.getParame(*server.sendParame())
            client.train()
            server.getParame(*client.uploadParame())
            updated_basis = client.getBasis()
            client_basis_dict[i] = updated_basis
            logging.info(f"HERE {client_basis_dict[0][0][0]}")
            # print(client_basis_dict[0][0][0])
            client.setAllBasis(client_basis_dict)
        
        # The 2 step
        server.aggregate()
        # server.test(test_dataloader)

    
    # logging.info(f"--------------------Finally Test--------------------")
    logging.info(f"Basis: {FLAGS.basis}, LEARNING_RATE: {FLAGS.learning_rate}, Best Acc: {server.best_acc}")
    # test_acc = 0
    # for client in np.array(clients):
    #     client.getParame(*server.sendParame())
    #     test_acc += client.test()
    # test_acc /= len(clients)
    # logging.info("Clients Test Avg Acc: {:>8f}".format(test_acc))


# Single client run on local device
def SingleRunning(args, server, clients):

    print ("ENTERED")
    # for client in np.array(clients):
    #     client.getParame(*server.sendParame())

    for t in range(args.round):

        logging.info(f"---------------------Round {t}---------------------")
        for client in np.array(clients):
            if t!= 0:
                client.getParame(*server.sendParame())
            client.train()
            server.getParame(*client.uploadParame())
        
        server.aggregate()
    
    logging.info(f"--------------------Finally Test--------------------")
    # test_acc = 0
    # for client in np.array(clients):
    #     test_acc += client.test()
    # test_acc /= len(clients)
    # logging.info("Clients Test Avg Acc: {:>8f}".format(test_acc))


# Entire data will be run together
def EntireRunning(args):

    args.labeled_rate = 1.0
    args.num_client = 1

    # init all data on one client
    server, clients = init_fed(args)
    for t in range(args.round):

        logging.info(f"---------------------Round {t}---------------------")
        clients[0].train()
    
    logging.info(f"--------------------Finally Test--------------------")
    clients[0].test()
 

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # # data and path setting
    # parser.add_argument('--dataset', default='aifb', type=str)

    # parser.add_argument('--state_dir', '-state_dir', default='../log/state/', type=str)
    # parser.add_argument('--log_dir', '-log_dir', default='../log/', type=str)
    # parser.add_argument('--tb_log_dir', '-tb_log_dir', default='../log/tb_log/', type=str)
    # parser.add_argument('--labeled_rate', default=0.7, type=float)

    # # one task hyperparam
    # parser.add_argument('--lr', default=1e-2, type=int)
    # parser.add_argument('--l2norm', default=0, type=float, help="L2 norm coefficient")
    # parser.add_argument('--gpu', default='0', type=str, help="running on this device")
    # parser.add_argument('--num_cpu', default=1, type=int, help="number of cpu")
    # parser.add_argument('--run_mode', default='Fed', choices=['Fed',
    #                                                                 'Single',
    #                                                                 'Entire'], type=str)

    # # for Fed-RGCN
    # parser.add_argument('--num_client', default=10, type=int, help="number of clients")
    # parser.add_argument('--fraction', default=1, type=float, help="fractional clients")
    # parser.add_argument('--round', default=20, type=int, help="federated learning rounds")
    # parser.add_argument('--local_epoch', default=20, type=int, help="local epochs to train")
    # parser.add_argument('--n_hidden', default=16, type=float, help="number of hidden units")
    # parser.add_argument('--n_bases', default=100, type=int, help="use number of relations as number of bases")
    # # parser.add_argument('--n_bases', default=-1, type=int, help="use number of relations as number of bases")
    # parser.add_argument('--n_hidden_layers', default=2, type=int, help="use 1 input layer, 1 output layer, n hidden layers")

    # # for random 
    # # parser.add_argument('--seed', default=87532, type=int)
    # parser.add_argument('--seed', default=12345, type=int)

    # args = parser.parse_args()
    # args_str = json.dumps(vars(args))

    # args.gpu = torch.device('cuda:' + args.gpu)
    # args.gpu = torch.device('cpu')

    # set random seed
    # np.random.seed(args.seed)
    # # torch.manual_seed(args.seed)
    # # torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)
    # tf.set_random_seed(args.seed)

    # # create directories
    # init_dir(args)

    # init writer
    # writer = SummaryWriter(args.tb_log_dir + args.run_mode + "/")
    # args.writer = writer

    # init logger
    # init_logger(args)
    # logging.info(args_str) 

    # if args.run_mode == 'Fed':
        # init FL setting
        # server, clients = init_fed(args)
        # running
    # args = {"gpu"}
    args = 1
    server, clients, flags = classification(args)
    FedRunning(args, server, clients, flags)
    # elif args.run_mode == 'Single':
    #     # init FL setting
    #     # server, clients = init_fed(args)
    #     args.num_client = 1
    #     print("SINGLE_ENTERED")
    #     server, clients = classification(args)
    #     # running
    #     print("SINGLE RUNNING")
    #     SingleRunning(args, server, clients)
    # elif args.run_mode == 'Entire':
    #     EntireRunning(args)
