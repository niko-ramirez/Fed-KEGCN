# from dgl._deprecate.graph import DGLGraph
import tensorflow as tf
# import torch
# from torch import nn, optim
from tqdm import tqdm
# from dgl import DGLGraph
import copy
from metrics import *
from utils import *


class Client:
    def __init__(self, client_id, data, model, FLAGS, epoch, lr, l2norm, model_path, logging, eval_func) -> None:
        # client setting
        self.client_id = client_id
        self.model = model
        self.epoch = epoch
        self.flags = FLAGS
        self.logging = logging
        self.get_eval = eval_func
        self.train_data = data["train"]
        self.test_data = data["test"]
        self.valid = data["valid"]
        self.y = data["y"]
        self.support = data["support"]
        self.all_basis_for_loss = {}
        self.var_to_grad = {}
        self.server_grads_to_apply = []
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.33
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())


        self.Epoch = -1                    # record the FL round by self
        self.round = None                  # record the mask round of FL this round from the server
        self.val_acc = None
        self.model_param = None
        self.placeholders = model.placeholders

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        acc_best = 0.
        test_acc = 0.
        self.Epoch += 1

        self.var_to_grad = {}

        # Train model
        # self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.flags.epochs):
            # Construct feed dictionary
            feed_dict = construct_feed_dict(1.0, self.support, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.flags.dropout})

            # Training step

            new_grad = [(grad,variable) for grad,variable in self.model.opt_grads]
    
            # non_none_list = [(grad, variable.name) for grad,variable in new_grad if grad != None and variable.name.__contains__("ent_weights")]

            # new_none_list = []
            # for grad,variable in new_grad:
            #     if grad != None and variable.name.__contains__("client_"+ str(self.client_id)):
            #         parsed_name = variable.name.split("_")
            #         for grad2, var_name in non_none_list:
            #             parsed = var_name.split("_")
            #             key = int(parsed[3]) % 2
            #             if int(parsed_name[3]) % 2 == key:
            #                 new_none_list.append((grad2 + grad, var_name))

            # non_none_list = new_none_list

            # temp_new_grad = []
            # for grad,variable in new_grad:
            #     if grad != None and variable.name.__contains__("ent_weights"):
            #         for small_grad, var_name in non_none_list:
            #             if var_name == variable.name:
            #                 temp_new_grad.append((small_grad, variable))
            #     else:
            #         temp_new_grad.append((grad, variable))
                
            # new_grad = temp_new_grad


            # model_grad_for_server = []
            # for grad, var_name in non_none_list:
            #     if var_name in self.var_to_grad:
            #         sum_all_grads = grad + self.var_to_grad[var_name]
            #     else: 
            #         sum_all_grads = grad
            #     self.var_to_grad[var_name] = sum_all_grads.eval(feed_dict = feed_dict, session=self.sess)
            #     model_grad_for_server.append((sum_all_grads.eval(feed_dict = feed_dict, session=self.sess), var_name))
            # self.model.grads = [model_grad_for_server, self.model.num_nodes]

            total_layers = 2

            client_basis_var_to_grad = {}
            for grad,variable in new_grad:
                if grad != None and variable.name.__contains__("client_"+ str(self.client_id)):
                    print(variable.name)
                    parsed_name = variable.name.split("_")
                    client_basis_var_to_grad[int(parsed_name[3]) % total_layers] = grad


            model_grad_for_server = []
            temp_new_grad = []
            for grad,variable in new_grad:
                if grad != None and variable.name.__contains__("ent_weights"):
                    var_name = variable.name
                    # COMBINES OT LOSS GRADIENTS TO NORMAL LOSS
                    parsed_name = var_name.split("_")
                    key = int(parsed_name[3]) % total_layers
                    small_grad = grad + client_basis_var_to_grad[key]


                    # HANDLES LOGIC FOR ADDING ALL GRADIENTS OVER A LOCAL TRAIN
                    if var_name in self.var_to_grad and False:
                        sum_all_grads = small_grad + self.var_to_grad[var_name]
                    else: 
                        sum_all_grads = small_grad
                    self.var_to_grad[var_name] = sum_all_grads.eval(feed_dict = feed_dict, session=self.sess)
                    model_grad_for_server.append((sum_all_grads.eval(feed_dict = feed_dict, session=self.sess), var_name))

                    #HANDLES WHEN SERVER SENDS GRADIENTS TO APPLY
                    if epoch == 0 and self.server_grads_to_apply: 
                        if key == 0:
                            key = total_layers
                        small_grad = small_grad + self.server_grads_to_apply[key-1]

                    
                    temp_new_grad.append((small_grad, variable))

                else:
                    temp_new_grad.append((grad, variable))
                
            new_grad = temp_new_grad
            self.model.grads = [model_grad_for_server, self.model.num_nodes]
        
        
            # total_layers = 2
            # if epoch == 0 and self.server_grads_to_apply: 
            #     new_grads = []
            #     for grad,variable in new_grad:
            #         if grad != None and variable.name.__contains__("ent_weights"):
            #             parsed_name = variable.name.split("_")
            #             key = int(parsed_name[3]) % total_layers
            #             if key == 0:
            #                 key = total_layers
            #             grad_with_server = grad + self.server_grads_to_apply[key-1]
            #         else:
            #             grad_with_server = grad
            #         new_grads.append((grad_with_server, variable))
            #     new_grad = new_grads

            outputs = self.sess.run([self.model.opt_apply(new_grad), self.model.loss], feed_dict=feed_dict)

            # ORIGINAL CALL TO MINIMIZE, DONT CHANGE
            # outputs = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)

            # Print results
            # if epoch % 10 == 0:
                # self.logging.info("Epoch: {} train_loss= {:.5f}".format(epoch, outputs[-1]))

            if epoch % 4 == 0 and self.valid is not None:
                # model.evaluate()
                output_embeddings = self.sess.run(self.model.outputs, feed_dict=feed_dict)
                train_acc, _ = self.get_eval(output_embeddings[0], self.train_data, self.y, self.logging)
                # self.logging.info("Train Accuracy: %.3f" % (train_acc * 100))
                acc, _ = self.get_eval(output_embeddings[0], self.valid, self.y, self.logging)
                # self.logging.info("Valid Accuracy: %.3f" % (acc * 100))
            # Training step
            outputs = sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
            # Print results
            if epoch % 10 == 0:
                self.logging.info("Epoch: {} train_loss= {:.5f}".format(epoch+1, outputs[1]))

            if epoch % 10 == 0 and self.valid is not None:
                # model.evaluate()
                output_embeddings = sess.run(self.model.outputs, feed_dict=feed_dict)
                train_acc, _ = self.get_eval(output_embeddings[0], self.train_data, self.y, self.logging)
                self.logging.info("Train Accuracy: %.3f" % (train_acc * 100))
                acc, _ = self.get_eval(output_embeddings[0], self.valid, self.y, self.logging)
                self.logging.info("Valid Accuracy: %.3f" % (acc * 100))
                if acc > acc_best:
                    acc_best = acc
                    test_acc, result = self.get_eval(output_embeddings[0], self.test_data, self.y, self.logging)
                self.logging.info("Test Accuracy: %.3f" % (test_acc * 100))


            if epoch % 4 == 0 and epoch > 0 and self.valid is None:
                # model.evaluate()
                output_embeddings = self.sess.run(self.model.outputs, feed_dict=feed_dict)
                train_acc, _ = self.get_eval(output_embeddings[0], self.train_data, self.y, self.logging)
                # self.logging.info("Train Accuracy: %.3f" % (train_acc * 100))
                acc, temp = self.get_eval(output_embeddings[0], self.test_data, self.y, self.logging)
                self.logging.info(f"Client: {self.client_id}")
            if epoch % 10 == 0 and epoch > 0 and self.valid is None:
                # model.evaluate()
                output_embeddings = sess.run(self.model.outputs, feed_dict=feed_dict)
                train_acc, _ = self.get_eval(output_embeddings[0], self.train_data, self.y, self.logging)
                self.logging.info("Train Accuracy: %.3f" % (train_acc * 100))
                acc, temp = self.get_eval(output_embeddings[0], self.test_data, self.y, self.logging)
                self.logging.info("Test Accuracy: %.3f" % (acc * 100))
                if acc > acc_best:
                    acc_best = acc
                    result = temp


        # self.model_param = {}
        # for var_name, value in self.model.vars.items():
        #     if var_name.__contains__("ent_weights") and var_name.__contains__("Adam"):
        #         if var_name.__contains__("autorelgraphconvolution_1_vars"):
        #             self.model_param[(self.client_id, var_name)] = self.sess.run(value)
        #         elif var_name.__contains__("autorelgraphconvolution_2_vars"):
        #             self.model_param[(self.client_id, var_name)] = self.sess.run(value)

        # self.logging.info("Optimization Finished! Best Valid Acc: {} Test: {}".format(
        #                 round(acc * 100,2), " ".join([str(round(i*100,2)) for i in result])))


    def getGradParams(self, round, param):
        # self.round = round
        # print("COPYING: ", param)

        for grad, var in self.model.grads:
            model_var_name = var.name
            if model_var_name.__contains__("ent_weights") and model_var_name.__contains__("Adam"):
                if model_var_name.__contains__("autorelgraphconvolution_1_vars"):
                    self.model.opt_apply([(param[model_var_name], var)])
                
                elif model_var_name.__contains__("autorelgraphconvolution_2_vars"):
                    self.model.opt_apply([(param[model_var_name], var)])
        # copy_params = {}
        # for key, values in param.items():
        #     if key.__contains__("Adam"):
        #         print(key)
        #         print(values)
        #         self.model.opt_apply([values, key])
        #         copy_params[key] = tf.identity(values)
        # self.model_param = copy_params
        # self.model.load_state_dict(copy_params)

    def getBasis(self):
        basis_dict = {}
        for model_var_name, var in self.model.vars.items():
            if model_var_name.endswith('ent_weights_0:0'):
                parsed_name = model_var_name.split("_")
                # print(parsed_name)
                if len(parsed_name[3]) <= 2:
                    key = int(parsed_name[3]) % 2
                else:
                    key = int(parsed_name[2]) % 2
                if key == 0:
                    key = 2
                basis_dict[key-1] = var.eval(session=self.sess)
                # print(basis_dict[key-1])
        return basis_dict


    # def getParame(self, round, param):
    #     # self.round = round
    #     # print("COPYING: ", param)

    #     # for model_var_name, var in self.model.vars.items():
    #     #     if model_var_name.__contains__("ent_weights") and model_var_name.__contains__("Adam"):
    #     #         if model_var_name.__contains__("autorelgraphconvolution_1_vars"):
    #     #             self.model.opt_apply([(param[model_var_name], var)])
    #     #             # self.vars[model_var].assign(weight_basis["layer_one_basis"])
    #     #         elif model_var_name.__contains__("autorelgraphconvolution_2_vars"):
    #     #             self.model.opt_apply([(param[model_var_name], var)])
    #     #             # self.vars[model_var].assign(weight_basis["layer_two_basis"])
    #     # copy_params = self.model_param.copy()
    #     for var_name, values in param.items():
    #         if var_name.__contains__("ent_weights"):
    #             self.model.vars[var_name].assign(values)
    #             # copy_params[var_name] = tf.identity(values)
    #     # self.model_param = copy_params
    #     # self.model.load_state_dict(copy_params)


    def getParame(self, round, param):
        for var_name, new_basis in param.items():
            self.model.vars[var_name].assign(new_basis)
        return
    
    def getGrad(self, round, global_grad):
        self.server_grads_to_apply = global_grad
        return
        self.logging.info("Optimization Finished! Best Valid Acc: {} Test: {}".format(
                        round(acc * 100,2), " ".join([str(round(i*100,2)) for i in result])))
    

    def getParame(self, round, param):
        self.round = round
        if self.model_param is not None:
            for layer in self.model_param:
                if ("layers.0" in layer) or layer.endswith("w_comp"): 
                    param[layer] = copy.deepcopy(self.model_param[layer])
        self.model.load_state_dict(param)


    # # upload the local model's parameters to parameter server
    def uploadParame(self):
        result_grads = self.model.grads.copy()
        self.model.grads = None
        return self.round, result_grads, self.val_acc
        #new_model_grads =[] 
        #for grad, var_name in self.model.grads[0]:
        #    new_model_grads.append((grad.eval(session=self.sess), var_name))
        #return self.round, [new_model_grads, self.model.grads[1]], self.val_acc

    def setAllBasis(self, basis_dicts):
        for client_k in basis_dicts:
            for k, value in basis_dicts[client_k].items():
                # value = np.ones(value.shape, dtype=np.float32)
                self.sess.run(self.model.basis_dict[client_k][k].assign(tf.constant(value)))
                # self.model.top_level_sess.run(self.model.basis_dict[client_k][k].assign(tf.constant(value)))
                # self.model.self.top_level_sesssess.run(self.model.basis_dict[client_k][k].assign(tf.constant(value)))
        self.all_basis_for_loss = basis_dicts

    # def setAllBasis(self, basis_dicts):
    #     self.all_basis_for_loss = basis_dicts[0][0]
    #     # temp_dict = {}
    #     # for k, v in basis_dicts.items():
    #     #     temp_dict[k] = tf.constant(basis_dict[0])
    #     self.model.basis_dict = tf.constant(basis_dicts[0][0])
        # param = {}
        # for layer in self.model.layers:
        #     param[layer] = self.model_param[layer]
        # return self.round, param, self.val_acc





    # def __init__(self, client_id, data, model, device, epoch, lr, l2norm, model_path, logging, writer) -> None:
    #     # log setting
    #     self.model_path = model_path
    #     self.logging = logging
    #     self.writer = writer

    #     # client setting
    #     self.client_id = client_id
    #     self.device = device
    #     self.data = data
    #     self.model = model.to(self.device)
    #     self.epoch = epoch

    #     self.Epoch = -1                    # record the FL round by self
    #     self.round = None                  # record the mask round of FL this round from the server
    #     self.val_acc = None
    #     self.model_param = None
        
    #     # training setting
    #     self.loss_fn = nn.CrossEntropyLoss()
    #     self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2norm)

    #     # data setting
    #     # node data process
    #     self.labels = tf.convert_to_tensor(data.labels).view(-1).to(self.device)
    #     # self.labels = torch.from_numpy(data.labels).view(-1).to(self.device)
    #     train_idx = data.train_idx
    #     self.val_idx = train_idx[ : len(train_idx) // 5]
    #     self.train_idx = train_idx[len(train_idx) // 5 : ]
    #     self.test_idx = data.test_idx
    #     # edges data process
    #     # edge_type = torch.from_numpy(data.edge_type).type(torch.LongTensor)
    #     # edge_norm = torch.from_numpy(data.edge_norm).type(torch.LongTensor).unsqueeze(1)
    #     edge_type = tf.convert_to_tensor(data.edge_type)
    #     edge_norm = tf.convert_to_tensor(data.edge_norm).unsqueeze(1)
    #     # edge_type = torch.from_numpy(data.edge_type)
    #     # edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)
    #     # create graph
    #     self.graph = DGLGraph().to(self.device)
    #     self.graph.add_nodes(data.num_nodes)
    #     self.graph.add_edges(data.edge_src, data.edge_dst)
    #     self.graph.edata.update({'rel_type': edge_type.to(self.device), 'norm': edge_norm.to(self.device)})
    #     # self.graph.edata.update({'rel_type': edge_type.to(self.device).type(torch.LongTensor), 'norm': edge_norm.to(self.device).type(torch.LongTensor)})


    # def train(self):

    #     pbar = tqdm(range(self.epoch))
    #     self.val_acc = 0
    #     self.Epoch += 1
    #     for _ in pbar:
    #         # Compute prediction error
    #         logits = self.model(self.graph)
    #         loss = self.loss_fn(logits.type(torch.FloatTensor)[self.train_idx], self.labels.type(torch.LongTensor)[self.train_idx])

    #         # Backpropagation
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    #         # train_acc = (logits.type(torch.FloatTensor)[self.train_idx].argmax(1) == self.labels[self.train_idx]).float().sum()
    #         train_acc = (logits[self.train_idx].argmax(1) == self.labels[self.train_idx]).float().sum()
    #         train_acc = train_acc / len(self.train_idx)
    #         # val_acc =  (logits.type(torch.FloatTensor)[self.val_idx].argmax(1) == self.labels[self.val_idx]).float().sum()
    #         val_acc =  (logits[self.val_idx].argmax(1) == self.labels[self.val_idx]).float().sum()
    #         val_acc = val_acc / len(self.val_idx)

    #         self.val_acc = self.val_acc + val_acc

    #         pbar.set_description("Client {:>2} Training: Train Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}".format(
    #                             self.client_id, loss.item(), train_acc, val_acc))
    #         self.writer.add_scalar(f"training/loss/{self.client_id}", loss.item(), self.Epoch * self.epoch + _)
    #         self.writer.add_scalar(f"training/acc/{self.client_id}", train_acc, self.Epoch * self.epoch + _)
    #         self.writer.add_scalar(f"val/acc/{self.client_id}", val_acc, self.Epoch * self.epoch + _)

    #     self.writer.add_embedding(logits[self.val_idx], self.labels[self.val_idx], global_step=self.Epoch, tag="clent"+str(self.client_id))

    #     self.model_param = self.model.state_dict()
    #     self.val_acc = self.val_acc / self.epoch

    # def test(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         logits = self.model(self.graph)
    #         test_loss = self.loss_fn(logits.type(torch.FloatTensor)[self.test_idx], self.labels.type(torch.LongTensor)[self.test_idx])
    #         test_acc =  (logits[self.test_idx].argmax(1) == self.labels[self.test_idx]).float().sum()
    #         test_acc = test_acc / len(self.test_idx)

    #     self.logging.info("Client {:>2} Test: Test Loss: {:.4f} | Test Acc: {:.4f}".format(
    #         self.client_id, test_loss.item(), test_acc
    #     ))
    #     # save to disk
    #     torch.save(self.model_param, self.model_path + "client" + str(self.client_id) + '_model.ckpt')

    #     return test_acc
    
    # # get the global model's parameters from parameter server
    # def getParame(self, round, param):
    #     self.round = round
    #     if self.model_param is not None:
    #         for layer_name in self.model_param:
    #             if ("layers.0" in layer_name) or layer_name.endswith("w_comp"): 
    #                 param[layer_name] = copy.deepcopy(self.model_param[layer_name])
    #     self.model.load_state_dict(param)


    # # upload the local model's parameters to parameter server
    # def uploadParame(self):
    #     param = {}
    #     for layer_name in self.model_param:
    #         if not layer_name.endswith("w_comp") and ( "layers.0" not in layer_name):
    #             param[layer_name] = self.model_param[layer_name]
    #     return self.round, param, self.val_acc
