import pandas as pd
from abc import ABCMeta, abstractmethod

from ..source import SourceCode


class FeatureExtractor():
    __metaclass__ = ABCMeta

    def extract_header(self):
        return None

    @abstractmethod
    def extract_row(self, source):
        """Returns a list of features extracted from the source."""
        raise NotImplementedError()

    def extract_into_frame(self, sources):
        """Creates a dataframe from lists of data and an optional header."""
        rows = list(map(self.extract_row, sources))
        return pd.DataFrame(rows, columns=self.extract_header())

    def extract(self, sources):
        if isinstance(sources, SourceCode):
            sources = [sources]
        return self.extract_into_frame(sources) 
