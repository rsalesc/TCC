import argparse
import json
import pickle

import sklearn.metrics

import numpy as np
import matplotlib.pyplot as plt

LW = 2


def argparsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("--curves", nargs="+")
    parser.add_argument("--save-to", required=True)

    return parser.parse_args()


def load_curve(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def draw_curve(curve):
    auc = sklearn.metrics.auc(np.array(curve["far"]),
                              1.0 - np.array(curve["frr"]))
    label = "{} (auc = {:0.2f}, eer = {:0.2f})".format(
        curve["name"], auc, curve["eer"])
    plt.plot(np.array(curve["far"]),
             1.0 - np.array(curve["frr"]), lw=LW, label=label)


def main(args):
    curves = []
    for curve in args.curves:
        curves.append(load_curve(curve))

    plt.figure()

    for curve in curves:
        draw_curve(curve)

    plt.plot([0, 1], [0, 1], color="navy", lw=LW, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")
    plt.savefig(args.save_to, bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    args = argparsing()
    main(args)
