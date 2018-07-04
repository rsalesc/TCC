import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import (Input, Dense, Flatten, Lambda,
                                            Embedding)
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.optimizers import Adam

from .common import (contrastive_loss, l2_normalization, accuracy,
                     argmax_accuracy, euclidean_distance)
from .base import BaseModel


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
        
        self._contrastive_loss_fn = contrastive_loss(0.2)
        self._accuracy_fn = accuracy(0.0, 2.0, 40)
        self._argmax_accuracy_fn = argmax_accuracy(0.0, 2.0, 40)

    def input_shape(self):
        return (self._input_size, )

    def loader_objects(self):
        return {
            "tf": tf,
            "contrastive_loss": self._contrastive_loss_fn,
            "accuracy": self._accuracy_fn,
            "argmax_accuracy": self._argmax_accuracy_fn
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
        contrastive_loss = self._contrastive_loss_fn
        accuracy = self._accuracy_fn
        argmax_accuracy = self._argmax_accuracy_fn

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
        x = BatchNormalization()(x)

        x = self.FullyConnectedLayer(1024)(x)
        x = self.FullyConnectedLayer(256)(x)
        x = self.FullyConnectedLayer(
            self._output_size, activation=None, dropout=False, batch_norm=False)(x)
        x = Lambda(l2_normalization)(x)

        return Model(input, x)

    def ConvLayer(self,
                  filters,
                  kernel_size,
                  strides=1,
                  activation="relu",
                  dropout=True,
                  batch_norm=True):
        def builder(z):
            x = Convolution1D(
                filters, kernel_size, strides=strides,
                activation=activation)(z)
            if dropout:
                x = Dropout(self._dropout_conv)(x)
            if batch_norm:
                x = BatchNormalization()(x)

            return x

        return builder

    def FullyConnectedLayer(self, size, activation="relu", dropout=True, batch_norm=True):
        def builder(z):
            x = Dense(size, activation=activation)(z)
            if dropout:
                x = Dropout(self._dropout_fc)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            return x

        return builder
