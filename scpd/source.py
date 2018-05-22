import numpy as np
from .utils import accumulator_sample


class SourceCode():
    def __init__(self, author, code=None, path=None):
        self._author = author
        self._code = code
        self._path = path

    def author(self):
        return self._author

    def fetch(self):
        if self._code is not None:
            return self._code
        elif self._path is not None:
            return open(self._path).read()
        else:
            raise AssertionError('No source code')

    def __str__(self):
        sec = self._code if self._code is not None else self._path
        return '(%s, %s)' % (self._author, sec)


class SourceCodePairing():
    def _build_intervals(self, sources):
        self._intervals = {}
        last = 0
        for i in range(len(sources)):
            if (i + 1 == len(sources)
                    or sources[i].author() != sources[i + 1].author()):
                self._intervals[sources[i].author()] = (last, i+1)
                last = i+1

    def _get_equal_pairs(self, sources, K):
        acc = [0]
        authors = []
        for author, interval in self._intervals.items():
            n = interval[1] - interval[0]
            acc.append(acc[-1] + n*n)
            authors.append(author)

        samples = accumulator_sample(acc, K)
        res = []
        for i, rem in samples:
            interval = self._intervals[authors[i]]
            n = interval[1] - interval[0]
            a = rem // n + interval[0]
            b = rem % n + interval[0]
            res.append((sources[a], sources[b]))
        return res

    def _get_different_pairs(self, sources, K):
        q = len(sources)
        acc = [0]
        authors = []
        for author, interval in self._intervals.items():
            n = interval[1] - interval[0]
            acc.append(acc[-1] + n*(q-n))
            authors.append(author)
            if q - n == 0:
                raise AssertionError(
                    'sources should have at least two authors')

        samples = accumulator_sample(acc, K)
        res = []
        for i, rem in samples:
            interval = self._intervals[authors[i]]
            n = interval[1] - interval[0]
            a = rem // (q-n) + interval[0]
            b = rem % (q-n)
            if b >= interval[0]:
                b += interval[1] - interval[0]
            res.append((sources[a], sources[b]))
        return res

    def make_pairs(self, sources, k1, k2):
        self._build_intervals(sources)
        res = []
        res.extend(self._get_equal_pairs(sources, k1))
        res.extend(self._get_different_pairs(sources, k2))
        return res
