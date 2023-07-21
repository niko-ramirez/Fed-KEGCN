import tensorflow as tf
import numpy as np
import scipy.spatial
from scipy.sparse import isspmatrix
import ot
import sys

def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name+":0")
    except:
        return tf.placeholder(tf.int32, name=name)


def align_loss(embeddings, align_labels, gamma, num_negs, names_neg, mode="L1"):
    def loss_metric(X, Y):
        if mode == "L1":
            loss = tf.reduce_sum(tf.abs(X - Y), 1)
        elif mode == "sim":
            loss = tf.nn.sigmoid(-tf.reduce_sum(X * Y, 1))
        else:
            exit("wrong loss mode")
        return loss
    def get_ranking_loss(names, num_labels):
        neg_left = get_placeholder_by_name(names[0])
        neg_right = get_placeholder_by_name(names[1])
        neg_l_x = tf.nn.embedding_lookup(embeddings, neg_left)
        neg_r_x = tf.nn.embedding_lookup(embeddings, neg_right)
        neg_value = loss_metric(neg_l_x, neg_r_x)
        neg_value = - tf.reshape(neg_value, [num_labels, num_negs])
        loss_value = neg_value + tf.reshape(pos_value, [num_labels, 1])
        loss_value = tf.nn.relu(loss_value)
        return loss_value

    left_labels = align_labels[:, 0]
    right_labels = align_labels[:, 1]
    num_labels = len(align_labels)
    left_x = tf.nn.embedding_lookup(embeddings, left_labels)
    right_x = tf.nn.embedding_lookup(embeddings, right_labels)
    pos_value = loss_metric(left_x, right_x) + gamma

    loss_1 = get_ranking_loss(names_neg[:2], num_labels)
    loss_2 = get_ranking_loss(names_neg[2:], num_labels)

    final_loss = tf.reduce_sum(loss_1) + tf.reduce_sum(loss_2)
    final_loss /= (2.0 * num_negs * num_labels)

    return final_loss


def class_loss(embeddings, test, y):
    y_pre = tf.nn.embedding_lookup(embeddings, test)
    if isspmatrix(y):
        y_test = y[test].todense()
        y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    else:
        y_test = y[test]
        y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_test, logits=y_pre)
    return tf.reduce_mean(loss)

def fed_class_loss(embeddings, test, y, basis_dict, client_id, num_nodes, sess):
    y_pre = tf.nn.embedding_lookup(embeddings, test)
    if isspmatrix(y):
        y_test = y[test].todense()
        y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    else:
        y_test = y[test]
        y_true = tf.reshape(tf.argmax(y_test, 1), (1,-1))
    f_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_test, logits=y_pre)

    # mu is a hyperparameter, M is a cost matrix
    # getting OT between each client's basis for each layer
    mu = 10
    lamb = 10.0
    constant = mu/10
    ot_loss = tf.constant(0, dtype=tf.float32)
    # V_k_base = basis_dict[0][0][0]
    # V_j_base = basis_dict[1][0][0]

    # new_k_shape = (V_k_base.shape[0]*V_k_base.shape[1], 1)
    # new_v_shape = (V_j_base.shape[0]*V_j_base.shape[1], 1)
    # V_k_base = tf.reshape(V_k_base, new_k_shape)
    # V_j_base = tf.reshape(V_j_base, new_v_shape)
    # M = np.array([[float(i) for i in range(V_k_base.shape[0])] for _ in range(V_k_base.shape[0])])
    # for row in range(M.shape[0]):
    #     M[row] -= row
    # M = np.abs(M, dtype=np.float32)
    # ot_loss = sink(M, (new_k_shape[0], new_v_shape[0]), lamb)
    def sink(cost_matrix, m_size, reg, numItermax=1000, stopThr=1e-9):
        # we assume that no distances are null except those of the diagonal of distances

        new_a = tf.ones(shape=(m_size[0],), dtype=tf.float32)/ m_size[0].value
        a = tf.expand_dims(new_a, axis=1)  # (na, 1)
        new_b = tf.ones(shape=(m_size[1],), dtype=tf.float32) / m_size[1].value
        b = tf.expand_dims(new_b, axis=1)  # (nb, 1)

        # init data
        Nini = m_size[0].value
        Nfin = m_size[1].value

        u = tf.expand_dims(tf.ones(Nini) / Nini, axis=1)  # (na, 1)
        v = tf.expand_dims(tf.ones(Nfin) / Nfin, axis=1)  # (nb, 1)

        K = tf.exp(-cost_matrix / reg)  # (na, nb)

        Kp = (1.0 / a) * K  # (na, 1) * (na, nb) = (na, nb)

        cpt = tf.constant(0)
        err = tf.constant(1.0)

        c = lambda cpt, u, v, err: tf.logical_and(cpt < numItermax, err > stopThr)

        def err_f1():
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = u * (K * tf.squeeze(v))  # (na, 1) * ((na, nb) * (nb,)) = (na, nb)
            err_ = tf.pow(tf.norm(tf.reduce_sum(transp) - b, ord=1), 2)  # (,)
            return err_

        def err_f2():
            return err

        def loop_func(cpt, u, v, err):
            KtransposeU = tf.matmul(tf.transpose(K, (1, 0)), u)  # (nb, na) x (na, 1) = (nb, 1)
            v = tf.div(b, KtransposeU)  # (nb, 1)
            u = 1.0 / tf.matmul(Kp, v)  # (na, 1)

            err = tf.cond(tf.equal(cpt % 10, 0), err_f1, err_f2)

            cpt = tf.add(cpt, 1)
            return cpt, u, v, err

        _, u, v, _ = tf.while_loop(c, loop_func, loop_vars=[cpt, u, v, err])

        # _, u, v, _ = tf.while_loop(c, loop_func, loop_vars=[cpt, V_k_base, V_j_base, err])

        result = tf.reduce_sum(u * K * tf.reshape(v, (1, -1)) * cost_matrix)

        return result


    def dmat(x, y):
        row_norms_A = tf.reduce_sum(tf.square(x), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(y), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        squared_dist = row_norms_A - 2 * tf.matmul(x, y, transpose_b=True) + row_norms_B
        dist = tf.sqrt(squared_dist + 1e-6)
        return dist
    
    # M = np.array([[i for i in range(32*32)] for _ in range(32*32)], dtype=np.float32)
    # for row in range(M.shape[0]):
    #     M[row] -= row
    # M = np.abs(M)
    # print(M)
    # M = tf.convert_to_tensor(M, dtype=tf.float32)
    for j in range(3):
        if j == client_id:
            continue
        for k, V_k in basis_dict[client_id].items():
            print(k)
            print(V_k.shape)
            V_j = basis_dict[j][k]
            # V_k = V_k.eval(session=sess)
            # V_j = V_j.eval(session=sess)

            for base in range(V_k.shape[0]):
                #V_k_base = V_k[base]
                #V_j_base = V_j[base]
            #     for smaller_base in range(V_k_base.shape[0]):
            #         V_k_base_small = V_k_base[:, smaller_base:smaller_base+1]
            #         V_j_base_small = V_j_base[:, smaller_base:smaller_base+1]
                # eps = np.zeros(V_k_base.shape)
                # eps += (1e-8)
                # V_k_base += eps
                # V_j_base += eps
                # M = ot.dist(x.reshape((V_k.shape[0].value, 1)), x.reshape((V_j.shape[0].value, 1)))
                new_k_shape = (V_k[base].shape[0]*V_k[base].shape[1], 1)
                V_k_base = tf.reshape(V_k[base], new_k_shape)
                V_j_base = tf.reshape(V_j[base], new_k_shape)
                #index_array = np.arange(0, len(V_k_base), dtype=int)
      
                #index_array = np.reshape(index_array, (len(V_k_base), 1))
                #M = ot.dist(index_array, index_array)
                # M = np.array([[i for i in range(V_k_base.shape[0])] for _ in range(V_k_base.shape[0])], dtype=np.float32)
                # for row in range(M.shape[0]):
                #     M[row] -= row
                # M = np.abs(M)

                M = dmat(V_k_base, V_j_base)
                # # ot_loss += sess.run([sink], feed_dict={x_tf:V_k_base, y_tf:V_j_base})
                ot_loss += sink(M, (new_k_shape[0], new_k_shape[0]), lamb) 
                # M = np.reshape(M, (new_k_shape[0]*new_k_shape[0],1))
                # M = ot.dist(V_k_base, V_j_base)
                #M /= M.max()
                # sess = tf.Session()
                # sess.run(tf.global_variables_initializer())
                # V_k = V_k.eval(session=sess)
                # V_j = V_j.eval(session=sess)
                # V_k = tf.make_ndarray(tf.make_tensor_proto(tf.constant(tf.identity(V_k))))
                # V_j = tf.make_ndarray(tf.make_tensor_proto(tf.constant(tf.identity(V_j))))
                # ot_loss += tf.constant(ot.sinkhorn2(V_k_base, V_j_base, M, lamb), dtype=tf.float32)
                # ot_loss = tf.Print(ot_loss, ["updated ot_loss is ", ot_loss])
    ot_loss = constant*ot_loss
    loss = f_loss + ot_loss
    #loss = tf.Print(loss, ["f_loss is ", f_loss, "ot_loss is ", ot_loss, "V_k_basis is" , V_k_base])
    return tf.reduce_mean(loss)

def link_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def label_loss(embeddings, test, y):
    y_pre = tf.nn.embedding_lookup(embeddings, test)
    if isspmatrix(y):
        y_test = y[test].todense()
    else:
        y_test = y[test]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test, logits=y_pre)
    return tf.reduce_mean(loss)

def get_class(embeddings, test, y, logging):
    y_pre = embeddings[test]
    if isspmatrix(y):
        y_test = y[test].todense()
        y_true = np.argmax(y_test, 1).reshape(1,-1)[0]
    else:
        y_test = y[test]
        y_true = np.argmax(y_test, 1).reshape(1,-1)
    y_pre = np.argmax(y_pre, 1).reshape(1,-1)
    # print(np.concatenate([y_true, y_pre]))
    correct_prediction = np.equal(y_pre, y_true)
    acc = np.mean(correct_prediction)
    return acc, [acc]

def get_label(embeddings, test, y, logging):
    y_pre = embeddings[test]
    if isspmatrix(y):
        y_test = y.todense()[test]
        y_test = np.squeeze(np.asarray(y_test))
    else:
        y_test = y[test]
    y_pre = np.argsort(-y_pre, 1)
    result_list = []

    for K in [1,3,5]:
        precision = 0
        NDCG = 0
        y_pre_K = y_pre[:, :K]
        coeff = 1./np.log(np.arange(1,K+1) + 1)
        for i,each in enumerate(y_pre_K):
            if np.sum(y_test[i]) <= 0:
                continue
            precision += np.sum(y_test[i, each])/K
            DCG_i = np.sum(y_test[i, each]*coeff)
            norm = np.sum(1./np.log(np.arange(1,min(K, np.sum(y_test[i]))+1) + 1))
            NDCG_i = DCG_i/norm
            NDCG += NDCG_i
        precision = precision/len(y_pre_K)
        NDCG = NDCG/len(y_pre_K)
        logging.info("Classification Precision %d: %.3f" % (K, precision * 100))
        logging.info("Classification NDCG %d: %.3f" % (K, NDCG * 100))
        result_list.append(round(precision, 4))
        result_list.append(round(NDCG, 4))
    return result_list[4], result_list


def get_align(embeddings, test_pair, logging, metric="cityblock", K=(1, 5, 10, 50, 100)):
    def get_metrics(sim, pos=0):
        top_hits = [0] * len(K)
        mrr = 0
        for ind in range(sim.shape[pos]):
            rank_ind = np.where(sim[(slice(None),) * pos + (ind,)].argsort() == ind)[0][0]
            mrr += 1/(rank_ind+1)
            for j in range(len(K)):
                if rank_ind < K[j]:
                    top_hits[j] += 1
        return top_hits, mrr

    embeddings_left = np.array([embeddings[e1] for e1, _ in test_pair])
    embeddings_right = np.array([embeddings[e2] for _, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(embeddings_left, embeddings_right, metric=metric)

    top_hits_l2r, mrr_l2r = get_metrics(sim, pos=0)
    top_hits_r2l, mrr_r2l = get_metrics(sim, pos=1)

    test_metric = ["Hits@{}".format(K[i]) for i in range(len(K))]
    test_metric = " ".join(test_metric)
    left_result = [top_hits_l2r[i] / len(test_pair) * 100 for i in range(len(K))]
    right_result = [top_hits_r2l[i] / len(test_pair) * 100 for i in range(len(K))]
    all_result = [(left_result[i] + right_result[i])/2 for i in range(len(right_result))]
    left_result = [str(round(i, 3)) for i in left_result]
    right_result = [str(round(i, 3)) for i in right_result]
    all_result = [str(round(i, 3)) for i in all_result]
    logging.info(test_metric)
    logging.info("l:\t" + "\t".join(left_result))
    logging.info("r:\t" + "\t".join(right_result))
    logging.info("a:\t" + "\t".join(all_result))
    logging.info('MRR-l: %.3f' % (mrr_l2r / len(test_pair)))
    logging.info('MRR-r: %.3f' % (mrr_r2l / len(test_pair)))
    logging.info('MRR-a: %.3f' % ((mrr_l2r+mrr_r2l)/2 / len(test_pair)))
