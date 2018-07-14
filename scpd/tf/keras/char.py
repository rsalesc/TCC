import tensorflow as tf
from keras import backend as K
from keras.utils import Sequence
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Flatten, Lambda, Embedding)
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import BatchNormalization, Activation
from keras.layers import Dropout
from keras.optimizers import Adam

from .common import (contrastive_loss, l2_normalization, euclidean_distance,
                     triplet_loss)
from .base import BaseModel
from .metrics import TripletOnKerasMetric, ContrastiveOnKerasMetric


class SimilarityCharCNN(BaseModel):
    def __init__(self,
                 input_size,
                 alphabet_size,
                 embedding_size,
                 output_size,
                 dropout_conv=0.5,
                 dropout_fc=0.5,
                 margin=0.2,
                 metric="accuracy",
                 metric_margin=None,
                 optimizer=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert optimizer is not None
        if not isinstance(metric, list):
            metric = [metric]
        self._input_size = input_size
        self._alphabet_size = alphabet_size
        self._dropout_conv = dropout_conv
        self._dropout_fc = dropout_fc
        self._embedding_size = embedding_size
        self._output_size = output_size
        self._optimizer = optimizer

        self._margin = margin
        self._contrastive_loss_fn = contrastive_loss(margin)
        self._metric = list(map(lambda x: ContrastiveOnKerasMetric(
                            metric_margin or margin, metric=x), metric))

    def input_shape(self):
        return (self._input_size, )

    def loader_objects(self):
        res = {"tf": tf, "contrastive_loss": self._contrastive_loss_fn}
        for metric in self._metric:
            res[metric.__name__] = metric
        return res

    def build(self):
        x1 = Input(shape=self.input_shape())
        x2 = Input(shape=self.input_shape())
        siamesis = self.SiamesisNetwork()
        a = siamesis(x1)
        b = siamesis(x2)

        distance = Lambda(euclidean_distance)([a, b])

        self.model = Model([x1, x2], distance)

    def compile(self):
        contrastive_loss = self._contrastive_loss_fn

        self.model.compile(
            loss=contrastive_loss,
            optimizer=self._optimizer,
            metrics=self._metric)

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
            self._output_size,
            activation=None,
            dropout=False,
            batch_norm=False)(x)
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

    def FullyConnectedLayer(self,
                            size,
                            activation="relu",
                            dropout=True,
                            batch_norm=True):
        def builder(z):
            x = Dense(size, activation=activation)(z)
            if dropout:
                x = Dropout(self._dropout_fc)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            return x

        return builder


class TripletCharCNN(SimilarityCharCNN):
    def __init__(self, *args, metric="accuracy", metric_margin=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(metric, list):
            metric = [metric]

        self._triplet_loss_fn = triplet_loss(self._margin)
        self._metric = list(map(lambda x: TripletOnKerasMetric(
                            metric_margin or self._margin, metric=x), metric))

    def loader_objects(self):
        res = {"tf": tf, "triplet_loss": self._triplet_loss_fn}

        for metric in self._metric:
            res[metric.__name__] = metric
        return res

    def compile(self):
        triplet_loss = self._triplet_loss_fn

        self.model.compile(
            loss=triplet_loss, optimizer=self._optimizer, metrics=self._metric)

    def build(self):
        x = Input(shape=self.input_shape())
        embeddings = self.SiamesisNetwork()(x)
        identity = Activation("linear", name="output_embedding")(embeddings)

        self.model = Model(x, identity)

    def embeddings_to_watch(self):
        return ["output_embedding"]
