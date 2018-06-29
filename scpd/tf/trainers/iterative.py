import numpy as np

from tqdm import tqdm

from .base import BaseTrainer


class IterativeTrainer(BaseTrainer):
    def __init__(self,
                 sess,
                 model,
                 provider,
                 iters_per_epoch=1,
                 batch_size=1,
                 *args,
                 **kwargs):
        super().__init__(sess, model, provider, *args, **kwargs)
        self._iters_per_epoch = iters_per_epoch
        self._batch_size = batch_size

    def validate(self):
        pass

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
        print("Loss: {}, model acc: {}".format(loss, acc))
