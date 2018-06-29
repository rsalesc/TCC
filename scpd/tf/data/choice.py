import numpy as np

from .base import BaseBatchProvider


class RandomBatchProvider(BaseBatchProvider):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if x.shape[0] != y.shape[0]:
            raise AssertionError(
                "x/y must contain the same number of observations")
        self._x = x
        self._y = y

    def next_batch(self, batch_size):
        indices = np.random.choice(self._x.shape[0], batch_size)
        return self._x[indices], self._y[indices]
