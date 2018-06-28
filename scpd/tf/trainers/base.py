import tensorflow as tf


class BaseTrainer:
    def __init__(self, sess, model, provider, n_epochs=1, *args, **kwargs):
        self._config = kwargs
        self._n_epochs = n_epochs
        self._sess = sess
        self._model = model
        self._provider = provider

        with tf.variable_scope("global_step"):
            self._global_step_tensor = tf.Variable(
                0, trainable=False, name="global_step")
            self._model.set_global_step(self._global_step_tensor)

        # init tensorflow session
        self._init = tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer())
        self._sess.run(self._init)

    def train(self):
        for current_epoch in range(
                self._model.epoch(self._sess), self._n_epochs + 1):
            self.train_epoch()
            self._sess.run(self._model.step_epoch)

    def train_epoch(self):
        raise NotImplementedError()
