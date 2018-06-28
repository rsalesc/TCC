import tensorflow as tf

from .base import BaseModel


class SiamesisMLPModel(BaseModel):
    def __init__(self, input_size, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_size = input_size
        self._hidden_size = hidden_size

    def compile(self):
        self.x1 = tf.placeholder(tf.float32, shape=[None, self._input_size])
        self.x2 = tf.placeholder(tf.float32, shape=[None, self._input_size])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        with tf.variable_scope("siamesis") as scope:
            self._d1 = self._inner_layer(self.x1, "inner1")
            scope.reuse_variables()
            self._d2 = self._inner_layer(self.x2, "inner2")

        self._concat = tf.concat([self._d1, self._d2], axis=-1)
        self._fc1 = tf.layers.dense(
            self._concat,
            self._hidden_size,
            activation=tf.nn.sigmoid,
            name="fc1")
        self._fc2 = tf.layers.dense(self._concat, 2, name="fc2")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y, logits=self._fc2))
            self.optimize = tf.train.AdamOptimizer(
                self._learning_rate).minimize(
                    self.cross_entropy, global_step=self._global_step_tensor)

            # accuracy
            correct_mask = tf.equal(
                tf.argmax(self._fc2, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    def _inner_layer(self, input_tensor, name):
        return tf.layers.dense(
            input_tensor,
            self._hidden_size,
            activation=tf.nn.sigmoid,
            name=name)
