import argparse
import pickle
import numpy as np
import random
import json

from keras.utils import Sequence

from scpd.utils import extract_labels, opens
from scpd.tf.keras.metrics import CompletePairContrastiveScorer

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

import tensorflow as tf
import keras

import matplotlib.pyplot as plt

import constants
import nn
import dataset
from nn import *

from itertools import cycle
CYCOL = cycle([
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
])


class SequenceWrapper(Sequence):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __len__(self):
        return len(self._wrapped)

    def __getitem__(self, idx):
        return self._wrapped[idx]


def get_dummy_optimizer():
    return None


def argparsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-to", default=None)
    parser.add_argument("--threshold-granularity", type=int, default=256)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-file", nargs="+", required=True)
    parser.add_argument("--caide", default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--experiment", choices=["roc", "embedding", "knn", "top-n"],
                        default="roc")

    parser.add_argument("--dataset", choices=[
        "cf", "gcj", "gcjt", "gcjr", "cf+gcj"
    ], default="cf")
    parser.add_argument("--roc-name", default="classifier")

    parser.add_argument("--neighbors", default=6, type=int)
    parser.add_argument("--average", type=int, default=1)

    subparsers = parser.add_subparsers(title="models", dest="model")
    subparsers.required = True

    # MLP
    mlp = subparsers.add_parser("mlp")

    mlp.add_argument("--pca", type=int, default=None)
    mlp.add_argument("--verbose", default=False, action="store_true")
    mlp.add_argument("--select", type=int, default=None)

    # Char CNN
    cnn = subparsers.add_parser("cnn")

    # Code LSTM
    lstm = subparsers.add_parser("lstm")
    lstm.set_defaults(get_nn=lambda x:
                      nn.get_triplet_lstm_nn(x, get_dummy_optimizer()))
    lstm.set_defaults(infer_fn=lstm_embedding_infer_batches)

    smax = subparsers.add_parser("softmax_lstm")
    smax.set_defaults(get_nn=lambda x:
                      nn.get_softmax_lstm_nn(x, get_dummy_optimizer()))
    smax.set_defaults(infer_fn=lstm_logits_infer_batches)
    smax.add_argument("--top", nargs="+", type=int, default=[1, 3])
    smax.set_defaults(experiment="top-n")

    # Code KNN+LSTM
    lstm = subparsers.add_parser("knn_lstm")
    lstm.set_defaults(experiment="knn")
    lstm.set_defaults(get_nn=lambda x:
                      nn.get_triplet_lstm_nn(x, get_dummy_optimizer()))
    lstm.set_defaults(infer_fn=lstm_knn_infer)

    return parser.parse_args()


def lstm_logits_infer_batches(args):
    xargs = load_xargs(args.model_path)
    net = load_nn(args)
    sources = load_dataset(args)

    input_size = (xargs.max_lines, xargs.max_chars)
    extract_fn = (nn.extract_hierarchical_features if args.average == 1
                  else nn.HierarchicalWindowFeatureExtractor(0.0))

    hot_labels = extract_labels([sources], one_hot=True)[0]
    labels = extract_labels([sources])[0]

    sequence = (
        nn.CategoricalCodeSequence(sources,
                                   hot_labels,
                                   input_size=input_size,
                                   batch_size=args.batch_size,
                                   fn=extract_fn))

    y_pred = []
    for i in range(len(sequence)):
        x, y_true = sequence[i]
        acc = 0.0
        for j in range(args.average):
            acc += net.model.predict_on_batch(x)
        y_pred.append(acc / args.average)

    sequence = (
        nn.CategoricalCodeSequence(sources,
                                   labels,
                                   input_size=input_size,
                                   batch_size=args.batch_size,
                                   fn=extract_fn))
    return sequence, y_pred


def lstm_knn_infer(args):
    xargs = load_xargs(args.model_path)
    net = load_nn(args)
    training_sources, test_sources, _ = load_knn_dataset(args)

    input_size = (xargs.max_lines, xargs.max_chars)
    extract_fn = (nn.extract_hierarchical_features if args.average == 1
                  else nn.HierarchicalWindowFeatureExtractor(0.0))

    training_labels, test_labels = extract_labels([
        training_sources, test_sources])

    base = [(training_sources, training_labels),
            (test_sources, test_labels)]

    res = []
    for sources, labels, in base:
        sequence = (
            nn.CategoricalCodeSequence(sources,
                                       labels,
                                       input_size=input_size,
                                       batch_size=args.batch_size,
                                       fn=extract_fn))

        labels = []
        y_pred = []
        for i in range(len(sequence)):
            x, y_true = sequence[i]
            acc = 0.0
            for j in range(args.average):
                acc += net.model.predict_on_batch(x)
            y_pred.extend(acc / args.average)
            labels.extend(y_true)

        res.append((np.array(y_pred), np.array(labels)))

    return res


def lstm_embedding_infer_batches(args):
    xargs = load_xargs(args.model_path)
    net = load_nn(args)
    sources = load_dataset(args)

    input_size = (xargs.max_lines, xargs.max_chars)
    extract_fn = (nn.extract_hierarchical_features if args.average == 1
                  else nn.HierarchicalWindowFeatureExtractor(0.0))

    labels = extract_labels([sources])[0]
    sequence = (
        nn.CategoricalCodeSequence(sources,
                                   labels,
                                   input_size=input_size,
                                   batch_size=args.batch_size,
                                   fn=extract_fn))

    y_pred = []
    for i in range(len(sequence)):
        x, y_true = sequence[i]
        acc = 0.0
        for j in range(args.average):
            acc += net.model.predict_on_batch(x)
        y_pred.append(acc / args.average)

    return sequence, y_pred


def load_knn_dataset(args):
    random.seed(constants.MAGICAL_SEED)
    return dataset.preloaded_gcj_easiest(args.test_file,
                                         caide=args.caide)


def load_dataset(args, ds=None, files=None):
    if ds is None:
        ds = args.dataset
    if files is None:
        files = args.test_file

    random.seed(constants.MAGICAL_SEED)
    if ds == "cf":
        data = dataset.preloaded(files, caide=args.caide)
        return [x for y in data for x in y]
    if ds == "gcj":
        return dataset.preloaded_gcj_easiest(
            files, caide=args.caide)[0]
    if ds == "gcjt":
        return dataset.preloaded_gcj_easiest(
            files, caide=args.caide)[1]
    if ds == "gcjr":
        data = dataset.preloaded_gcj_random(
            files, caide=args.caide)
        return [x for y in data for x in y]
    if ds == "cf+gcj":
        cf = []
        gcj = []

        for path in files:
            if "gcj" in path:
                gcj.append(path)
            else:
                cf.append(path)

        res = []

        if len(gcj) > 0:
            res.extend(load_dataset(args, ds="gcjr", files=gcj))
        if len(cf) > 0:
            res.extend(load_dataset(args, ds="cf", files=cf))

        return res

    raise NotImplementedError()


def load_xargs(model_path):
    with open("{}.args.pkl".format(model_path), "rb") as f:
        return pickle.load(f)


def load_nn(args):
    xargs = load_xargs(args.model_path)
    model_h5 = "{}.h5".format(args.model_path)

    net = args.get_nn(xargs)
    nn.build_scpd_model(net, path=model_h5)
    print(net.model.summary())

    return net


def get_roc_steps(args):
    return np.linspace(0.0, 2.0, args.threshold_granularity)


def run_roc_experiment(args, infer_batches):
    assert args.save_to is not None

    scorer = CompletePairContrastiveScorer(get_roc_steps(args))
    sequence, a = infer_batches

    for i in range(len(sequence)):
        _, y_true = sequence[i]
        y_pred = a[i]

        scorer.handle(y_true, y_pred)

    eer = scorer.result("eer")
    print("EER : {}".format(eer))
    far = scorer.result("far")
    frr = scorer.result("frr")

    path = "{}.roc.pkl".format(args.save_to)
    with opens(path, "wb") as f:
        pickle.dump({
            "name": args.roc_name,
            "frr": frr,
            "far": far,
            "eer": eer
        }, f)


def run_embedding_experiment(args, infer_batches):
    assert args.save_to is not None

    sequence, a = infer_batches

    every = []
    by_label = {}

    for i in range(len(sequence)):
        _, y_true = sequence[i]
        y_pred = a[i]

        for j in range(len(y_true)):
            if y_true[j] not in by_label:
                by_label[y_true[j]] = []
            by_label[y_true[j]].append(len(every))
            every.append(y_pred[j])

    embeddings = TSNE(n_components=2).fit_transform(every)

    plt.figure()
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    for label, indices in by_label.items():
        c = next(CYCOL)
        x = []
        y = []
        for index in indices:
            x.append(embeddings[index][0])
            y.append(embeddings[index][1])

        plt.scatter(x, y, c=c)

    path = "{}.emb.png".format(args.save_to)
    plt.savefig(path, bbox_inches='tight', dpi=300)


def run_knn_experiment(args, infer_sets):
    training_set, test_set = infer_sets

    training_embeddings, training_labels = training_set
    nei = None

    if args.neighbors == 0:
        nei = NearestCentroid()
    else:
        nei = KNeighborsClassifier(n_neighbors=args.neighbors,
                                   weights="distance")

    print("Fitting nearest neighbor")
    nei.fit(training_embeddings, training_labels)

    test_embeddings, test_labels = test_set

    test_pred = nei.predict(test_embeddings)

    acc = np.mean(np.equal(test_pred, test_labels))
    print("Accuracy: {}".format(acc))


def run_top_experiment(args, infer_batches):
    sequence, a = infer_batches
    n_batches = len(sequence)

    results = {}
    for K in args.top:
        results[K] = 0
    total = 0

    for i in range(n_batches):
        _, y_true = sequence[i]
        y_pred = np.array(a[i])

        total += len(y_true)

        for K in args.top:
            top_k_indices = np.argpartition(-y_pred, K-1)[..., :K]
            corr = np.sum(np.equal(top_k_indices,
                                   y_true[..., np.newaxis]), axis=-1) > 0
            results[K] += np.sum(corr)

    for K in args.top:
        print("top-{}: {}".format(K, 1.0 * results[K] / total))


def run_experiment(args, infer_batches):
    if args.experiment == "top-n":
        run_top_experiment(args, infer_batches)
    elif args.experiment == "roc":
        run_roc_experiment(args, infer_batches)
    elif args.experiment == "embedding":
        run_embedding_experiment(args, infer_batches)
    elif args.experiment == "knn":
        run_knn_experiment(args, infer_batches)
    else:
        raise NotImplementedError()


def main(args):
    run_experiment(args, args.infer_fn(args))


if __name__ == "__main__":
    args = argparsing()
    main(args)
