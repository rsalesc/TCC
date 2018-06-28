import os
import random
import shutil
import pandas as pd


def opens(*args, **kwargs):
    if len(args) == 0:
        raise AssertionError("File name argument should be informed")
    dirname = os.path.dirname(args[0])
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    return open(*args, **kwargs)

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
