import tensorflow as tf
import os


def load_latest(sess, saver, folder):
    if folder is None:
        return False
    checkpoint = tf.train.latest_checkpoint(folder)
    if checkpoint is not None:
        saver.restore(sess, checkpoint)
        return True
    return False


def init_variables(sess):
    # init tensorflow session
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    sess.run(init)


def get_global_step_tensor():
    return tf.Variable(0, trainable=False, name="global_step")


class SmartSaver:
    def __init__(self,
                 sess,
                 folder,
                 max_to_keep=5,
                 epoch_step=1,
                 verbose=False):
        self._sess = sess
        self._saver = tf.train.Saver(max_to_keep=max_to_keep)
        self._folder = os.path.abspath(folder)
        self._each = epoch_step
        self._verbose = verbose
        self._loaded = False

    def has_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = load_latest(self._sess, self._saver, self._folder)

    def save(self, global_step_tensor=None):
        if self._verbose:
            print("Saving session in {}...".format(self._folder))
        path = os.path.join(self._folder, "flow.ckpt")
        self._saver.save(self._sess, path, global_step_tensor)

    def should_save(self, epoch):
        return (epoch % self._each == 0)
