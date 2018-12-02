import tensorflow as tf
from keras import backend as K
from tensorflow.contrib.losses import metric_learning


def contrastive_loss(margin):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    def contrastive_loss(y_true, y_pred):
        return 0.5 * K.mean(y_true * K.square(y_pred) + (1 - y_true) *
                            K.square(K.maximum(margin - y_pred, 0)))

    return contrastive_loss


def triplet_loss(margin):
    def triplet_loss(labels, embeddings):
        return metric_learning.triplet_semihard_loss(
            tf.reshape(labels, [-1]), embeddings, margin=margin)

    return triplet_loss


def categorical_loss(labels, logits):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)


def l2_normalization(x):
    return tf.nn.l2_normalize(x, axis=1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
