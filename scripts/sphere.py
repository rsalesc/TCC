import math
import numpy as np


def gen(n):
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    return v


def monte_carlo(n, th, its=100000):
    anchor = gen(n)
    ok = 0
    for i in range(its):
        if np.linalg.norm(anchor - gen(n)) < th:
            ok += 1

    assert ok > 0
    return float(its) / ok


def volume(n, r=None):
    if r is not None:
        return r**n * volume(n)
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0
    return 2.0 * math.pi / n * volume(n - 2)


def area(n, r=None):
    if n == 0:
        return 2.0
    return 2.0 * math.pi * volume(n - 1)
