import numpy as np
import ot
import matplotlib.pylab as plt

import tensorflow as tf

def sink_tf(M, m_size, reg, numItermax=1000, stopThr=1e-9):
    # we assume that no distances are null except those of the diagonal of distances

    a = tf.expand_dims(tf.ones(shape=(m_size[0],)) / m_size[0], axis=1)  # (na, 1)
    b = tf.expand_dims(tf.ones(shape=(m_size[1],)) / m_size[1], axis=1)  # (nb, 1)

    # init data
    Nini = m_size[0]
    Nfin = m_size[1]

    u = tf.expand_dims(tf.ones(Nini) / Nini, axis=1)  # (na, 1)
    v = tf.expand_dims(tf.ones(Nfin) / Nfin, axis=1)  # (nb, 1)

    K = tf.exp(-M / reg)  # (na, nb)

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

    result = tf.reduce_sum(u * K * tf.reshape(v, (1, -1)) * M)

    return result


def dmat_tf(x, y):
    """
    :param x: (na, 2)
    :param y: (nb, 2)
    :return:
    """
    mmp1 = tf.tile(tf.expand_dims(x, axis=1), [1, y.shape[0], 1])  # (na, nb, 2)
    mmp2 = tf.tile(tf.expand_dims(y, axis=0), [x.shape[0], 1, 1])  # (na, nb, 2)

    mm = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(mmp1, mmp2)), axis=2))  # (na, nb)

    print(mm)
    
    return mm

def distance(x, y):
    row_norms_A = tf.reduce_sum(tf.square(x), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(y), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    squared_dist = row_norms_A - 2 * tf.matmul(x, y, transpose_b=True) + row_norms_B
    dist = tf.sqrt(squared_dist + 1e-6)
    return dist

def main():
    na = 30
    nb = 30
    reg = 0.5

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -.8], [-.8, 1]])
    x_tf = tf.placeholder(dtype=tf.float32, shape=[na, 1])
    y_tf = tf.placeholder(dtype=tf.float32, shape=[nb, 1])
    # M_tf = dmat_tf(x_tf, y_tf)
    M_tf = distance(x_tf, y_tf)
    tf_sinkhorn_loss = sink_tf(M_tf, (na, nb), reg)

    print("I can compute the gradient for a", tf.gradients(tf_sinkhorn_loss, x_tf))
    print("I can compute the gradient for b", tf.gradients(tf_sinkhorn_loss, y_tf))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    xs = ot.datasets.make_2D_samples_gauss(na, mu_s, cov_s)[:,0:1]
    xt = ot.datasets.make_2D_samples_gauss(nb, mu_t, cov_t)[:,0:1]

    print(xs.shape)
    print(xt.shape)
    # Visualization
    # plt.figure(1)
    # plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    # plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    # plt.legend(loc=0)
    # plt.title('Source and target distributions')
    # plt.show()

    # TF - sinkhorn
    tf_sinkhorn_loss_val = sess.run(tf_sinkhorn_loss, feed_dict={x_tf: xs, y_tf: xt})
    print(' tf_sinkhorn_loss', tf_sinkhorn_loss_val)

    # POT - sinkhorn
    M = ot.dist(xs.copy(), xt.copy(), metric='euclidean')
    a = np.ones((na,)) / na
    b = np.ones((nb,)) / nb  # uniform distribution on samples
    pot_sinkhorn_loss = ot.sinkhorn2(a, b, M, reg)
    print('pot_sinkhorn_loss', pot_sinkhorn_loss)


if __name__ == '__main__':
    main()