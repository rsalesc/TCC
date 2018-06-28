import tensorflow as tf

from .data import RandomBatchProvider
from .model import SiamesisMLPModel
from .trainer import IterativeTrainer


class SiamesisMLPTrainer(IterativeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self):
        x, y = self._provider.next_batch(self._batch_size)
        feed_dict = {}
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
        input_size = training_features.shape[1]
        model = SiamesisMLPModel(input_size, 10)
        model.compile()
        if args.train:
            provider = RandomBatchProvider(training_features, test_features)
            trainer = SiamesisMLPTrainer(
                sess,
                model,
                provider,
                iters_per_epoch=10,
                n_epochs=10,
                batch_size=10)
            trainer.train()
