import tensorflow as tf
from tf_slim import layers
import tensorflow.keras.layers as k


def combine_embeddings_dense(embeddings, denses):
    """
    combine embeddings and dense tensors
    :param embeddings: batch_size * dimension
    :param denses: batch_size * dimension
    :return: batch_size * dimension
    """
    embedding = tf.concat(embeddings, 1)
    dense = tf.concat(denses, 1)
    return tf.concat([embedding, dense], 1)


def fm(embeddings):
    """
    fm output of embeddings
    :param embeddings: list of batch_size * dimension
    :return:
    """
    embeddings = tf.concat(embeddings, 1)
    sum_sqrt = tf.square(tf.reduce_sum(embeddings, 1))  # sum(x) ^ 2
    sqrt_sum = tf.reduce_sum(tf.square(embeddings), 1)  # sum(x ^ 2)
    value = 0.5 * tf.subtract(sum_sqrt, sqrt_sum)
    value = tf.expand_dims(value, [1])
    linear = k.Dense(1, name='linear_output')(embeddings)
    return tf.concat([value, linear], -1)


def deep_fm(embeddings, denses):
    """
    deep fm code
    :param embeddings:
    :param denses:
    :return:
    """
    inputs = combine_embeddings_dense(embeddings, denses)
    deep_layer = layers.stack(inputs, layers.fully_connected, [128, 64, 32], scope='fc')
    fm_layer = fm(embeddings)
    last_layer = tf.concat([deep_layer, fm_layer], 1)
    logits = k.Dense(units=1, name='logits')(last_layer)
    return logits



