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
from .common import (contrastive_loss, l2_normalization, euclidean_distance)


class SimilarityMLP(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout=0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._dropout = dropout

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
        x = input

        x = self.FullyConnectedLayer(self._hidden_size, dropout=True)(x)
        x = self.FullyConnectedLayer(self._hidden_size // 2)(x)
        x = self.FullyConnectedLayer(self._output_size, activation=None)(x)
        x = Lambda(l2_normalization)(x)

        return Model(input, x)

    def FullyConnectedLayer(self, size, activation="relu", dropout=False):
        def builder(z):
            x = Dense(size, activation=activation)(z)
            if dropout:
                x = Dropout(self._dropout)(x)
            return x

        return builder
