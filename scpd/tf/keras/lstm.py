import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (Input, Dense, LSTM, Lambda,
                                            Embedding)
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional

from .common import (l2_normalization, triplet_loss)
from .base import BaseModel
from .metrics import TripletOnKerasMetric


class TripletLineLSTM(BaseModel):
    def __init__(self,
                 alphabet_size,
                 embedding_size,
                 output_size,
                 line_capacity,
                 char_capacity,
                 dropout_line=0.0,
                 dropout_char=0.0,
                 margin=0.2,
                 metric="accuracy",
                 metric_margin=None,
                 optimizer=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert optimizer is not None
        self._alphabet_size = alphabet_size
        self._embedding_size = embedding_size
        self._output_size = output_size
        self._line_capacity = line_capacity
        self._char_capacity = char_capacity
        self._optimizer = optimizer
        self._dropout_line = dropout_line
        self._dropout_char = dropout_char

        self._margin = margin
        self._triplet_loss_fn = triplet_loss(margin)
        self._metric = TripletOnKerasMetric(
            metric_margin or margin, metric=metric)

    def input_shape(self):
        # lines x chars
        return (None, None)

    def loader_objects(self):
        return {
            "tf": tf,
            "triplet_loss": self._triplet_loss_fn,
            self._metric.__name__: self._metric
        }

    def build(self):
        x = Input(shape=self.input_shape())
        embeddings = self.SiamesisNetwork()(x)

        self.model = Model(x, embeddings)

    def compile(self):
        triplet_loss = self._triplet_loss_fn

        self.model.compile(
            loss=triplet_loss,
            optimizer=self._optimizer,
            metrics=[self._metric])

    def SiamesisNetwork(self):
        input = Input(shape=self.input_shape())
        x = Embedding(self._alphabet_size + 1, self._embedding_size)(input)

        x = TimeDistributed(
            LSTM(self._char_capacity, dropout=self._dropout_char))(x)
        x = Bidirectional(
            LSTM(self._line_capacity, dropout=self._line_capacity))(x)
        x = Dense(self._output_size, activation=None)(x)
        x = Lambda(l2_normalization)(x)

        return Model(input, x)
