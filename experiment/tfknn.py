import tensorflow as tf

from EBH.utility.const import projectroot
from EBH.utility.operation import load_dataset


def build_tfgraph():

    npX, npY = load_dataset(as_matrix=True, optimalish=True)

    test_point = tf.placeholder(dtype="float64", shape=[1, 20*4])

    X = tf.get_variable("X", initializer=tf.constant(npX))
    Y = tf.get_variable("Y", initializer=tf.constant(npY))
    d = tf.norm(X - test_point, axis=1)
    dval, darg = tf.arg_min(d, dimension=0)[:5]
    votes = Y[darg]

    n0 = tf.reduce_sum(tf.equal(votes, 0))
    n1 = tf.reduce_sum(tf.equal(votes, 1))
    n2 = tf.reduce_sum(tf.equal(votes, 2))

    pred = tf.argmax([n0, n1, n2])

    saver = tf.train.Saver([X, Y, pred])
    sess = tf.InteractiveSession()
    saver.save(sess, save_path=projectroot + "KNN.tfm")

    return pred


predop, newXop, newYop = build_tfgraph()

