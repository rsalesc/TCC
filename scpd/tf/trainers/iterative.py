import numpy as np

from tqdm import tqdm

from .base import BaseTrainer


class IterativeTrainer(BaseTrainer):
    def __init__(self, iters_per_epoch, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iters_per_epoch = iters_per_epoch
        self._batch_size = batch_size

    def train_epoch(self):
        loop = tqdm(
            range(self._iters_per_epoch),
            total=self._iters_per_epoch,
            desc="Training on epoch {}...".format(
                self._model.epoch(self._sess)))

        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(acc)

        # change to logger, save
        print("Loss: {}, accuracy: {}".format(loss, acc))
