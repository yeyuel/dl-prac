import tensorflow as tf
from tf_slim import layers
import tensorflow.keras.layers as k


def dcn(embeddings, denses, x0_len):
    def build_cross_layers(x0, x0_len, num_layers):
        def cross_layer(_x0, _x0_len, x, name):
            with tf.variable_scope(name):
                w = tf.get_variable('weight', [_x0_len], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b = tf.get_variable('bias', [_x0_len], initializer=tf.truncated_normal_initializer(stddev=0.01))
                xb = tf.tensordot(tf.reshape(-1, 1, _x0_len), w, 1)
                return x0 * xb + b + x
        x = x0
        for i in range(num_layers):
            x = cross_layer(x0, x0_len, x, 'cross_{}'.format(i))
        return x

    denses = tf.concat(denses, 1, 'dense')
    x0 = tf.concat(embeddings + [denses], 1)
    deep_layer = layers.stack(x0, layers.fully_connected, [x0_len, x0_len, x0_len], scope='fc')
    cross_layer = build_cross_layers(x0, x0_len, 2)
    last_layer = tf.concat([cross_layer, deep_layer], 1)
    logits = k.Dense(units=1, name='logits')(last_layer)
    return logits

