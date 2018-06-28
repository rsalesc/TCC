import tensorflow as tf
import numpy as np

from .data import RandomBatchProvider
from .model import SiamesisMLPModel
from .trainer import IterativeTrainer


class SiamesisMLPTrainer(IterativeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self):
        x, y = self._provider.next_batch(self._batch_size)
        x1, x2, *_ = np.hsplit(x, 2)
        feed_dict = {self._model.x1: x1, self._model.x2: x2, self._model.y: y}
        _, loss, acc = self._sess.run(
            [
                self._model.optimize, self._model.cross_entropy,
                self._model.accuracy
            ],
            feed_dict=feed_dict)
        return loss, acc


def execute(args,
            training_features=None,
            test_features=None,
            training_labels=None,
            test_labels=None):
    with tf.Session() as sess:
        # improve this
        input_size = training_features.shape[1] // 2
        model = SiamesisMLPModel(input_size, 10)
        model.compile()
        if args.train:
            provider = RandomBatchProvider(training_features, training_labels)
            trainer = SiamesisMLPTrainer(
                sess,
                model,
                provider,
                iters_per_epoch=10,
                n_epochs=10,
                batch_size=10)
            trainer.train()
