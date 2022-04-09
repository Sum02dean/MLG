import tensorflow as tf

from scipy.stats import spearmanr


def spearman_rankcor(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                       tf.cast(y_true, tf.float32)], Tout=tf.float32))

