import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.contrib.losses import metric_learning


def upper_triangular_flat(A):
    ones = tf.ones_like(A)
    mask_a = tf.matrix_band_part(ones, 0, -1)
    mask_b = tf.matrix_band_part(ones, 0, 0)
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)

    return tf.boolean_mask(A, mask)


def pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean 
                 distance matrix. 
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + tf.expand_dims(square_norm, 0)

    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def contrastive_loss(margin):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    def contrastive_loss(y_true, y_pred):
        return 0.5 * K.mean((1 - y_true) * K.square(y_pred) +
                            y_true * K.square(K.maximum(margin - y_pred, 0)))

    return contrastive_loss


def triplet_loss(margin):
    def triplet_loss(labels, embeddings):
        return metric_learning.triplet_semihard_loss(
            tf.reshape(labels, [-1]), embeddings, margin=margin)

    return triplet_loss


def l2_normalization(x):
    return tf.nn.l2_normalize(x, axis=1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def accuracy_per_threshold(y_true, y_pred, thresholds):
    th = tf.reshape(thresholds, [1, -1])
    reshaped_pred = tf.reshape(y_pred, [-1, 1])
    reshaped_true = tf.reshape(y_true, [-1, 1])
    correct_matrix = K.equal(
        K.cast(reshaped_pred > th, y_true.dtype), reshaped_true)
    return K.mean(tf.cast(correct_matrix, tf.float32), axis=0)


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


def triplet_accuracy_per_threshold(labels, embeddings, thresholds):
    dist = pairwise_distances(embeddings)
    labels = tf.reshape(labels, [-1, 1])
    pair_labels = tf.cast(tf.not_equal(labels, tf.transpose(labels)), tf.int32)

    return accuracy_per_threshold(
        upper_triangular_flat(pair_labels), upper_triangular_flat(dist),
        thresholds)


def triplet_accuracy(min=0.0, max=2.0, steps=40):
    def triplet_accuracy(y_true, y_pred):
        thresholds = tf.linspace(min, max, steps)
        return K.max(
            triplet_accuracy_per_threshold(y_true, y_pred, thresholds))

    return triplet_accuracy


def triplet_argmax_accuracy(min=0.0, max=2.0, steps=40):
    def triplet_argmax_accuracy(y_true, y_pred):
        thresholds = tf.linspace(min, max, steps)
        idx = tf.argmax(
            triplet_accuracy_per_threshold(y_true, y_pred, thresholds))
        return tf.gather(thresholds, idx)

    return triplet_argmax_accuracy
