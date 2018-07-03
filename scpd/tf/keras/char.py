import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import (Input, Dense, Flatten, Lambda,
                                            Embedding)
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.optimizers import Adam

from .base import BaseModel


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.2
    return 0.5 * K.mean((1 - y_true) * K.square(y_pred) +
                        y_true * K.square(K.maximum(margin - y_pred, 0)))


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


def accuracy(y_true, y_pred):
    thresholds = tf.linspace(0.0, 2.0, 40)
    return K.max(accuracy_per_threshold(y_true, y_pred, thresholds))


def argmax_accuracy(y_true, y_pred):
    thresholds = tf.linspace(0.0, 2.0, 40)
    idx = tf.argmax(accuracy_per_threshold(y_true, y_pred, thresholds))
    return tf.gather(thresholds, idx)


class SimilarityCharCNN(BaseModel):
    def __init__(self,
                 input_size,
                 alphabet_size,
                 embedding_size,
                 output_size,
                 dropout_conv=0.5,
                 dropout_fc=0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._input_size = input_size
        self._alphabet_size = alphabet_size
        self._dropout_conv = dropout_conv
        self._dropout_fc = dropout_fc
        self._embedding_size = embedding_size
        self._output_size = output_size

    def input_shape(self):
        return (self._input_size, )

    @staticmethod
    def loader_objects():
        return {
            "tf": tf,
            "contrastive_loss": contrastive_loss,
            "accuracy": accuracy,
            "argmax_accuracy": argmax_accuracy
        }

    def build(self):
        x1 = Input(shape=self.input_shape())
        x2 = Input(shape=self.input_shape())
        siamesis = self.SiamesisNetwork()
        a = siamesis(x1)
        b = siamesis(x2)

        distance = Lambda(euclidean_distance)([a, b])

        self.model = Model([x1, x2], distance)

    def compile(self, base_lr):
        optimizer = Adam(lr=base_lr)
        self.model.compile(
            loss=contrastive_loss,
            optimizer=optimizer,
            metrics=[accuracy, argmax_accuracy])

    def SiamesisNetwork(self):
        input = Input(shape=self.input_shape())
        x = Embedding(
            self._alphabet_size + 1,
            self._embedding_size,
            input_length=self._input_size)(input)

        x = self.ConvLayer(256, 7)(x)
        x = MaxPooling1D(3)(x)

        x = self.ConvLayer(256, 7)(x)
        x = MaxPooling1D(3)(x)

        x = self.ConvLayer(256, 3)(x)
        x = self.ConvLayer(256, 3)(x)
        x = self.ConvLayer(256, 3)(x)

        x = self.ConvLayer(256, 3)(x)
        x = MaxPooling1D(3)(x)

        x = Flatten()(x)

        x = self.FullyConnectedLayer(1024)(x)
        x = self.FullyConnectedLayer(256, dropout=False)(x)
        x = self.FullyConnectedLayer(
            self._output_size, activation=None, dropout=False)(x)
        x = Lambda(l2_normalization)(x)

        return Model(input, x)

    def ConvLayer(self,
                  filters,
                  kernel_size,
                  strides=1,
                  activation="relu",
                  dropout=True):
        def builder(z):
            x = Convolution1D(
                filters, kernel_size, strides=strides,
                activation=activation)(z)
            if dropout:
                x = Dropout(self._dropout_conv)(x)
            return x

        return builder

    def FullyConnectedLayer(self, size, activation="relu", dropout=True):
        def builder(z):
            x = Dense(size, activation=activation)(z)
            if dropout:
                x = Dropout(self._dropout_fc)(x)
            return x

        return builder
