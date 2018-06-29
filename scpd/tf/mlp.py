import tensorflow as tf
import numpy as np

from . import utils
from .utils import SmartSaver
from .data import RandomBatchProvider
from .models import BaseModel
from .trainers import IterativeTrainer


class SiamesisMLPModel(BaseModel):
    def __init__(self, input_size, hidden_size, learning_rate, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate

    def compile(self):
        self.x1 = tf.placeholder(tf.float32, shape=[None, self._input_size])
        self.x2 = tf.placeholder(tf.float32, shape=[None, self._input_size])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        with tf.variable_scope("siamesis") as scope:
            self._d1 = self._inner_layer(self.x1)
            scope.reuse_variables()
            self._d2 = self._inner_layer(self.x2)

        with tf.name_scope("loss"):
            self.loss = self._contrastive_loss(self._d1, self._d2, self.y)
            self.optimize = tf.train.AdamOptimizer(
                self._learning_rate).minimize(
                    self.loss, global_step=self._global_step_tensor)

        with tf.name_scope("inference"):
            self.inference = tf.norm(self._d1 - self._d2, axis=-1)

    def _contrastive_loss(self, i1, i2, y):
        y = y[:, 1]
        m = 0.2
        dw = tf.reduce_sum(tf.square(i1 - i2), axis=1)
        dw_sqrt = tf.sqrt(dw + 1e-7)
        loss = y * tf.square(tf.maximum(0.0, m - dw_sqrt)) + (1.0 - y) * dw
        return 0.5 * tf.reduce_mean(loss)

    def _inner_layer(self, input_tensor):
        fc1 = tf.layers.dense(
            input_tensor,
            self._hidden_size,
            activation=tf.nn.relu,
            name="dense1")
        fc2 = tf.layers.dense(
            fc1, self._hidden_size // 2, activation=tf.nn.relu, name="dense2")
        return tf.layers.dense(fc2, 8, name="dense_last")


class ValidationBatchProvider(RandomBatchProvider):
    def __init__(self, x, y, val_x, val_y, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        self._val_x = val_x
        self._val_y = val_y

    def validation(self):
        return self._val_x, self._val_y


class TwoInputTrainer(IterativeTrainer):
    def __init__(self,
                 sess,
                 model,
                 provider,
                 val_batch_size=1,
                 *args,
                 **kwargs):
        super().__init__(sess, model, provider, *args, **kwargs)
        self._val_batch_size = val_batch_size

    def train_step(self):
        # get batch and reshape it as needed
        x, y = self._provider.next_batch(self._batch_size)
        real_size = x.shape[0]
        x1, x2, *_ = np.split(x.reshape((real_size, -1)), 2, axis=1)

        # setup step
        feed_dict = {self._model.x1: x1, self._model.x2: x2, self._model.y: y}
        _, loss = self._sess.run(
            [
                self._model.optimize,
                self._model.loss,
            ], feed_dict=feed_dict)
        return loss, 0

    def infer(self, x1, x2):
        feed_dict = {self._model.x1: x1, self._model.x2: x2}
        return self._sess.run(self._model.inference, feed_dict=feed_dict)

    def validate(self):
        validation_tuple = self._provider.validation()
        if validation_tuple is None:
            return None
        validation_data, validation_labels = validation_tuple
        thresholds = np.linspace(0, 1, 20).reshape((1, -1))
        pred = []

        # infer in batches
        for i in range(0, validation_data.shape[0], self._val_batch_size):
            batch = validation_data[i:i + self._val_batch_size]
            a, b = np.split(batch, 2, axis=1)
            a = a.reshape((batch.shape[0], -1))
            b = b.reshape((batch.shape[0], -1))
            pred.extend(self.infer(a, b))

        pred_per_threshold = (np.array(pred).reshape(
            (-1, 1)) > thresholds).astype(int)
        correct_per_threshold = np.equal(
            pred_per_threshold, validation_labels[:, 1].reshape(-1, 1))
        acc = np.mean(correct_per_threshold, axis=0)
        argmax = np.argmax(acc)

        print("Validation acc: {}, threshold used: {}".format(
            acc[argmax], thresholds[0, argmax]))


def execute(args,
            training_features=None,
            test_features=None,
            training_labels=None,
            test_labels=None):
    with tf.Session() as sess:
        global_step_tensor = utils.get_global_step_tensor()

        # get number of features per instance
        input_size = training_features.shape[2]

        model = SiamesisMLPModel(input_size, 40, learning_rate=0.00005)
        model.compile()

        saver = None
        if args.checkpoint is not None:
            saver = SmartSaver(
                sess, args.checkpoint, epoch_step=5, verbose=True)
            if args.load:
                saver.load()

        if args.train:
            provider = ValidationBatchProvider(
                training_features,
                training_labels,
                val_x=test_features,
                val_y=test_labels)
            trainer = TwoInputTrainer(
                sess,
                model,
                provider,
                saver=saver,
                global_step_tensor=global_step_tensor,
                iters_per_epoch=500,
                n_epochs=1000,
                batch_size=64,
                val_batch_size=64)

            if not saver.has_loaded():
                utils.init_variables(sess)
            trainer.train()
