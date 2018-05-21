import os
import random


def opens(*args, **kwargs):
    if len(args) == 0:
        raise AssertionError("File name argument should be informed")
    os.makedirs(os.path.dirname(args[0]), exist_ok=True)
    return open(*args, **kwargs)


def sample_list(L, K):
    if K > len(L):
        raise AssertionError("Sample size cant be bigger than list")
    indices = random.sample(range(len(L)), K)
    indices.sort()
    return [L[i] for i in indices]
