import os
import random
import shutil
import pandas as pd


def opens(encoding="utf-8", *args, **kwargs):
    if len(args) == 0:
        raise AssertionError("File name argument should be informed")
    dirname = os.path.dirname(args[0])
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    return open(encoding=encoding, *args, **kwargs)


def copies(src, dst):
    dirname = os.path.dirname(dst)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    shutil.copyfile(src, dst)


def sample_list(L, K):
    if K > len(L):
        raise AssertionError("Sample size cant be bigger than list")
    indices = random.sample(range(len(L)), K)
    indices.sort()
    return [L[i] for i in indices]


def accumulator_sample(acc, K):
    if len(acc) == 0 or acc[0] != 0:
        raise AssertionError("accumulator should have first element 0")
    t = acc[-1]
    if K > t:
        print(K, t)
        raise AssertionError("size of sample greater than candidates")
    indices = random.sample(range(t), K)
    indices.sort()
    ptr = 0
    res = []
    for i in range(len(acc)):
        if ptr >= K:
            break
        while ptr < K and indices[ptr] < acc[i]:
            res.append((i - 1, indices[ptr] - acc[i - 1]))
            ptr += 1
    return res


def split_label(df):
    if not isinstance(df, pd.DataFrame):
        return df, None
    if "label" not in df:
        return df.values, None
    feature_matrix = df.drop(columns=["label"]).values
    label_array = df["label"].values
    return feature_matrix, label_array


def list_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def isiterable(iterable):
    try:
        iter(iterable)
        return True
    except TypeError:
        return False


class ObjectPairing():
    def diff_class(self, obj1, obj2):
        return self.get_class(obj1) != self.get_class(obj2)

    def get_class(self, obj):
        return obj.author()

    def _build_intervals(self, sources):
        self._intervals = {}
        last = 0
        for i in range(len(sources)):
            if (i + 1 == len(sources)
                    or self.diff_class(sources[i], sources[i + 1])):
                self._intervals[self.get_class(sources[i])] = (last, i + 1)
                last = i + 1

    def _get_equal_pairs(self, sources, K):
        acc = [0]
        authors = []
        for author, interval in self._intervals.items():
            n = interval[1] - interval[0]
            acc.append(acc[-1] + n * n)
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
            acc.append(acc[-1] + n * (q - n))
            authors.append(author)
            if q - n == 0:
                raise AssertionError(
                    'sources should have at least two classes')

        samples = accumulator_sample(acc, K)
        res = []
        for i, rem in samples:
            interval = self._intervals[authors[i]]
            n = interval[1] - interval[0]
            a = rem // (q - n) + interval[0]
            b = rem % (q - n)
            if b >= interval[0]:
                b += interval[1] - interval[0]
            res.append((sources[a], sources[b]))
        return res

    def make_pairs(self, objs, k1, k2):
        self._build_intervals(objs)
        res = []
        res.extend(self._get_equal_pairs(objs, k1))
        res.extend(self._get_different_pairs(objs, k2))
        return res
