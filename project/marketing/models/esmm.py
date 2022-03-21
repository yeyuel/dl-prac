import tensorflow as tf
from deepfm import deep_fm


def esmm(embeddings, denses):
    with tf.variable_scope('ctr_model'):
        ctr_logits = deep_fm(embeddings, denses)
    with tf.variable_scope('cvr_model'):
        cvr_logts = deep_fm(embeddings, denses)
    ctr_predictions = tf.sigmoid(ctr_logits, name='CTR')
    cvr_predicitons = tf.sigmoid(cvr_logts, name='CVR')
    ctcvr_predictions = tf.multiply(ctr_predictions, cvr_predicitons, name='CTCVR')

    # only ctr_predictions, ctcvr_predictions will participate in loss
    # cvr_predicitons participate in prediction
    return ctr_predictions, cvr_predicitons, ctcvr_predictions
