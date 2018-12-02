import argparse
import pickle
import numpy as np

from keras.optimizers import Adam
from keras.utils import Sequence

from scpd.utils import extract_labels
from scpd.tf.keras.metrics import CompletePairContrastiveScorer

import nn
import dataset
from nn import *


class SequenceWrapper(Sequence):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __len__(self):
        return len(self._wrapped)

    def __getitem__(self, idx):
        return self._wrapped[idx]


def get_dummy_optimizer():
    return Adam(lr=0.01)


def argparsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-to", default=None)
    parser.add_argument("--threshold-granularity", type=int, default=256)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--caide", default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--experiment", choices=["roc"], default="roc")
    parser.add_argument("--dataset", choices=["cf"], default="cf")

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

    return parser.parse_args()


def lstm_embedding_infer_batches(args):
    xargs = load_xargs(args.model_path)
    net = load_nn(args)
    sources = load_dataset(args)

    input_size = (xargs.max_lines, xargs.max_chars)
    extract_fn = nn.extract_hierarchical_features

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
        y_pred.append(net.model.predict_on_batch(x))

    return sequence, y_pred


def load_dataset(args):
    if args.dataset == "cf":
        return dataset.preloaded([args.test_file], caide=args.caide)[0]

    raise NotImplementedError()


def load_xargs(model_path):
    with open("{}.args.pkl".format(model_path), "rb") as f:
        return pickle.load(f)


def load_nn(args):
    xargs = load_xargs(args.model_path)
    model_h5 = "{}.h5".format(args.model_path)

    net = args.get_nn(xargs)
    nn.build_scpd_model(net, path=model_h5)
    net.compile()
    print(net.model.summary())

    return net


def get_roc_steps(args):
    return np.linspace(0.0, 2.0, args.threshold_granularity)


def run_roc_experiment(args, infer_batches):
    scorer = CompletePairContrastiveScorer(get_roc_steps(args))
    sequence, a = infer_batches

    for i in range(len(sequence)):
        _, y_true = sequence[i]
        y_pred = a[i]

        scorer.handle(y_true, y_pred)

    print(scorer.result("eer"))


def run_experiment(args, infer_batches):
    if args.experiment == "roc":
        run_roc_experiment(args, infer_batches)
    else:
        raise NotImplementedError()


def main(args):
    run_experiment(args, args.infer_fn(args))


if __name__ == "__main__":
    args = argparsing()
    main(args)
