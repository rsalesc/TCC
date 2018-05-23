import pandas as pd
from abc import ABCMeta, abstractmethod


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
