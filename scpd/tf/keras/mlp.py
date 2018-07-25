import tensorflow as tf
from keras import backend as K
from keras.utils import Sequence
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Flatten, Lambda, Embedding)
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Dropout, Activation
from keras.optimizers import Adam

from .base import BaseModel
from .common import (contrastive_loss, l2_normalization, euclidean_distance,
                     triplet_loss)
from .metrics import TripletOnKerasMetric


class TripletMLP(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 *args,
                 optimizer=None,
                 margin=0.2,
                 dropout=0.0,
                 metric="accuracy",
                 metric_margin=None,
                 **kwargs):
        super().__init__()
        if not isinstance(metric, list):
            metric = [metric]

        assert optimizer is not None

        self._optimizer = optimizer
        self._dropout = dropout
        self._hidden_size = hidden_size
        self._input_size = input_size
        self._output_size = output_size
        self._margin = margin
        self._triplet_loss_fn = triplet_loss(self._margin)
        self._metric = list(map(lambda x: TripletOnKerasMetric(
                            metric_margin or self._margin, metric=x), metric))

    def input_shape(self):
        return (self._input_size, )

    def loader_objects(self):
        res = {"tf": tf, "triplet_loss": self._triplet_loss_fn}

        for metric in self._metric:
            res[metric.__name__] = metric
        return res
    
    def SiamesisNetwork(self):
        input = Input(shape=self.input_shape())
        x = input

        for h in self._hidden_size[:-1]:
            x = self.FullyConnectedLayer(h, dropout=True)(x)
        x = self.FullyConnectedLayer(self._hidden_size[-1], activation=None)(x)
        x = Lambda(l2_normalization)(x)

        return Model(input, x)

    def FullyConnectedLayer(self, size, activation="relu", dropout=False):
        def builder(z):
            x = Dense(size, activation=activation)(z)
            if dropout:
                x = Dropout(self._dropout)(x)
            return x

        return builder

    def compile(self):
        triplet_loss = self._triplet_loss_fn

        self.model.compile(
            loss=triplet_loss, optimizer=self._optimizer, metrics=self._metric)

    def build(self):
        x = Input(shape=self.input_shape())
        embeddings = self.SiamesisNetwork()(x)
        identity = Activation("linear", name="output")(embeddings)

        self.model = Model(x, identity)

    def embeddings_to_watch(self):
        return ["output"]


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
