import tensorflow as tf


class BaseModel:
    def __init__(self, global_step_tensor=None, *args, **kwargs):
        self._global_step_tensor = global_step_tensor
        with tf.variable_scope("current_epoch"):
            self._current_epoch_tensor = tf.Variable(
                0, trainable=False, name="current_epoch")
            self.step_epoch = tf.assign(self._current_epoch_tensor,
                                        self._current_epoch_tensor + 1)

    def set_global_step(self, tensor):
        self._global_step_tensor = tensor

    def epoch(self, sess):
        return self._current_epoch_tensor.eval(sess)

    def compile(self):
        raise NotImplementedError()
