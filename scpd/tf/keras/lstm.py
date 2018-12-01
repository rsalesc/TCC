import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Dense, LSTM, Lambda, Embedding, Masking)
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Dropout, BatchNormalization, Activation
from keras.layers import RNN, LSTMCell

from .common import (l2_normalization, triplet_loss)
from .base import BaseModel
from .metrics import TripletOnKerasMetric


def select_cols(x, L):
    """Given 2D tensor `x` and 1D tensor `L` returns each x[i, L[i]] as 1D."""
    rows = tf.range(tf.shape(x)[0])
    indices = tf.stack([rows, L], axis=1)
    return tf.gather_nd(x, indices)


def reduce_dim(x, L):
    axis = tf.rank(L)
    initial_shape = tf.shape(x)
    final_shape = tf.concat([initial_shape[:axis], initial_shape[axis + 1:]],
                            0)

    xm = tf.reshape(x, tf.concat([[-1], initial_shape[axis:]], 0))
    Lm = tf.reshape(x, [-1])
    return tf.reshape(select_cols(xm, Lm), final_shape)


def gather_returns(tensor):
    x, L = tensor
    return reduce_dim(x, L - 1)


def has_gpu():
    return len(K.tensorflow_backend._get_available_gpus()) > 0


def mask_dropout(p):
    keep_prob = 1.0 - p

    def mask_dropout(x, mask):
        if mask is None:
            mask = tf.ones_like(x)
        noise = keep_prob + tf.random_uniform(tf.shape(mask))
        keep_mask = tf.cast(tf.floor(noise), tf.bool)
        return tf.logical_and(mask, keep_mask)
    return mask_dropout


def OptimizedLSTM(*args, **kwargs):
    return LSTM(*args, **kwargs)


class TripletLineLSTM(BaseModel):
    def __init__(self,
                 alphabet_size,
                 embedding_size,
                 output_size,
                 char_capacity,
                 line_capacity,
                 dropout_line=0.0,
                 dropout_char=0.0,
                 dropout_fc=0.0,
                 dropout_inter=0.0,
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
        self._line_capacity = line_capacity if isinstance(
            line_capacity, list) else [line_capacity]
        self._char_capacity = char_capacity if isinstance(
            char_capacity, list) else [char_capacity]
        self._optimizer = optimizer
        self._dropout_line = dropout_line
        self._dropout_char = dropout_char
        self._dropout_fc = dropout_fc
        self._dropout_inter = dropout_inter

        self._margin = margin
        self._triplet_loss_fn = triplet_loss(margin)

        self._metric = list(map(lambda x: TripletOnKerasMetric(
                            metric_margin or self._margin, metric=x), metric))

    def loader_objects(self):
        res = {"tf": tf, "triplet_loss": self._triplet_loss_fn}

        for metric in self._metric:
            res[metric.__name__] = metric
        return res

    def input_shape(self):
        # lines x chars
        return (None, None)

    def build(self):
        x = Input(shape=self.input_shape(), dtype="int32")
        embeddings = self.SiamesisNetwork()(x)
        identity = Activation("linear", name="output")(embeddings)

        self.model = Model(x, identity)

    def compile(self):
        triplet_loss = self._triplet_loss_fn

        self.model.compile(
            loss=triplet_loss, optimizer=self._optimizer, metrics=self._metric)

    def SiamesisNetwork(self):
        input = Input(shape=self.input_shape(), dtype="int32")

        x = TimeDistributed(
            Embedding(
                self._alphabet_size, self._embedding_size,
                mask_zero=True))(input)

        for i, cap in enumerate(self._char_capacity):
            is_last = (i + 1 == len(self._char_capacity))
            x = TimeDistributed(
                LSTM(cap, dropout=self._dropout_char, return_sequences=(not is_last)))(x)

        # get mask on original input and apply it to current output
        # (resets whatever mask is being propagated)
        x = Lambda(
            lambda x: x[0],
            mask=lambda y, _: Masking(mask_value=0).compute_mask(y[1]))(
                [x, input])

        # drop some lines
        # x = Lambda(lambda x: x, mask=mask_dropout(self._dropout_inter))(x)

        for i, cap in enumerate(self._line_capacity):
            is_last = (i + 1 == len(self._line_capacity))
            x = Bidirectional(
                LSTM(cap, dropout=self._dropout_line, return_sequences=(not is_last)))(x)

        # x = Dense(128, activation="relu")(x)
        # x = Dropout(self._dropout_fc)(x)
        # x = BatchNormalization()(x)

        x = Dense(self._output_size, activation=None)(x)
        x = Lambda(l2_normalization)(x)

        return Model(input, x)

    def embeddings_to_watch(self):
        return ["output"]
