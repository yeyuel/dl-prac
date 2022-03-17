import tensorflow as tf
from tf_slim import layers
import tensorflow.keras.layers as k


def xdeep_fm(embeddings, denses, emb_size, hiddens=[100, 50]):
    def compressed_interaction_layer(x0, xl, layer):
        d = x0.shape[1]
        m = x0.shape[-1]
        hpre = xl.shape[-1]
        inter = tf.matmul(tf.expand_dims(xl, [3]), tf.expand_dims(x0, [2]))  # B, D, hpre, m
        xcur = tf.reshape(inter, [-1, d, hpre * m])
        kernel = tf.get_variable(name='kernal{}'.format(str(layer)),
                                 shape=(hpre * m, layer))
        xcur = tf.matmul(xcur, kernel)
        xcur = tf.reshape(tf.transpose(xcur, [1, 0, 2]), [-1, d, layer])
        return xcur

    def build_cin(embeddings, layer_size):
        x0 = tf.stack(embeddings, 2)
        xcur = x0
        xhs = []
        for size in layer_size:
            xcur = compressed_interaction_layer(x0, xcur, size)
            xhs.append(tf.reduce_sum(xcur, 1))
        cin_out = tf.concat(xhs, 1)
        return cin_out

    dense = tf.concat(denses, 1, 'dense')
    dense_embeddings = k.Dense(units=emb_size, name='dense_embeddings')(dense)
    embeddings.append(dense_embeddings)
    # x0 = tf.concat(embeddings, 1)
    cin_out = build_cin(embeddings, hiddens)
    x0_len = len(embeddings) * emb_size
    deep_layer = layers.stack(dense, layers.fully_connected, [x0_len, x0_len], scope='fc')
    linear_output = k.Dense(units=1, name='linear_output')(dense)
    last_layer = tf.concat([cin_out, deep_layer, linear_output], 1)
    logits = k.Dense(units=1, name='logits')(last_layer)
    return logits