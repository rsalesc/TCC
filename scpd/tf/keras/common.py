import tensorflow as tf
from tensorflow.python.keras import backend as K


def contrastive_loss(margin):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    def contrastive_loss(y_true, y_pred):
        return 0.5 * K.mean((1 - y_true) * K.square(y_pred) +
                            y_true * K.square(K.maximum(margin - y_pred, 0)))
    return contrastive_loss


def l2_normalization(x):
    return tf.nn.l2_normalize(x, axis=1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def accuracy_per_threshold(y_true, y_pred, thresholds):
    th = tf.reshape(thresholds, [1, -1])
    reshaped_pred = tf.reshape(y_pred, [-1, 1])
    reshaped_true = tf.reshape(y_true, [-1, 1])
    correct_matrix = K.equal(
        K.cast(reshaped_pred > th, y_true.dtype), reshaped_true)
    return K.mean(K.cast(correct_matrix, y_true.dtype), axis=0)


def accuracy(min=0.0, max=2.0, steps=40):
    def accuracy(y_true, y_pred):
        thresholds = tf.linspace(min, max, steps)
        return K.max(accuracy_per_threshold(y_true, y_pred, thresholds))
    return accuracy


def argmax_accuracy(min=0.0, max=2.0, steps=40):
    def argmax_accuracy(y_true, y_pred):
        thresholds = tf.linspace(min, max, steps)
        idx = tf.argmax(accuracy_per_threshold(y_true, y_pred, thresholds))
        return tf.gather(thresholds, idx)
    return argmax_accuracy


