import pandas as pd
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from tqdm import tqdm

from .. import utils


class FeatureExtractor(metaclass=ABCMeta):
    def extract_header(self):
        return None

    @abstractmethod
    def extract_row(self, obj):
        """Returns a list of features extracted from the data."""
        raise NotImplementedError()

    def extract_into_frame(self, objs):
        """Creates rows of data and an optional header."""
        rows = list(map(self.extract_row, objs))
        return self.extract_header(), rows

    def extract(self, objs):
        """Returns a DataFrame of features from a list of objects."""
        header, rows = self.extract_into_frame(objs)
        return pd.DataFrame(rows, columns=header)


class BatchFeatureExtractor(FeatureExtractor):
    def __init__(self, extractor, batch_size=1):
        if not isinstance(extractor, FeatureExtractor):
            raise AssertionError(
                "BatchFeatureExtractor expects a FeatureExtractor")
        super().__init__()
        self._extractor = extractor
        self._batch_size = batch_size

    @contextmanager
    def bootstrap(self, objs):
        yield objs

    def extract_header(self):
        return self._extractor.extract_header()

    def extract_row(self, obj):
        return self._extractor.extract_row(obj)

    def extract_into_frame(self, objs, monitor=False):
        rows = []
        batch_list = utils.list_batch(objs, self._batch_size)
        batch_list_size = (
            len(objs) + self._batch_size - 1) // self._batch_size
        iterable = batch_list if not monitor else tqdm(
            batch_list, total=batch_list_size, desc='Batch Feature Extraction')
        for batch in iterable:
            with self.bootstrap(batch) as bootstrapped_batch:
                rows.extend(map(self.extract_row, bootstrapped_batch))
        return self.extract_header(), rows

    def extract(self, objs, monitor=False):
        header, rows = self.extract_into_frame(objs, monitor=monitor)
        return pd.DataFrame(rows, columns=header)

    def lazily_extract_into_frame(self, objs):
        for batch in utils.list_batch(objs, self._batch_size):
            extracted = None
            with self.bootstrap(batch) as bootstrapped_batch:
                extracted = self.extract_into_frame(bootstrapped_batch)
            yield extracted

    def lazily_extract(self, objs):
        for header, rows in self.lazily_extract_into_frame(objs):
            yield pd.DataFrame(rows, columns=header)
