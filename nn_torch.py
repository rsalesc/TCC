import argparse
import random

import numpy as np
import sys

import nn
import torch
from scpd.utils import LinearDecay, extract_labels
from scpd.tf.keras.metrics import CompletePairContrastiveScorer
from scpd.torch import lstm


def extract_hierarchical_features(source, input_size=None):
    assert isinstance(input_size, tuple) and len(input_size) == 2
    max_lines, max_chars = input_size
    assert max_lines is not None
    assert max_chars is not None
    lines = [np.array(nn.encode_text(line[:max_chars]))
             for line in source.fetch().splitlines()]
    lines = [(x if len(x) > 0 else np.array([0])) for x in lines]
    lines = lines[-max_lines:]

    return lines


def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=0, type=int)
    parser.add_argument("--max-epochs", default=1000, type=int)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--lr-decay", default=0, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--optimizer", default="adam",
                        choices=["adam", "rmsprop"])
    parser.add_argument("--save-to", default=".cache/torch")
    parser.add_argument("--no-checkpoint", action="store_true", default=False)
    parser.add_argument("--tensorboard-dir", default="/opt/tensorboard")
    parser.add_argument(
        "--reset-tensorboard", action="store_true", default=False)
    parser.add_argument("--threshold-granularity", type=int, default=256)

    parser.add_argument("--training-file", default=nn.TRAINING_DAT)
    parser.add_argument("--validation-file", default=nn.TEST_DAT)
    parser.add_argument("--embedding-file", default=nn.TEST_DAT)
    parser.add_argument("--embedding-period", type=int, default=3)
    parser.add_argument(
        "--download-dataset", action="store_true", default=False)
    parser.add_argument("--validation-batch-size", type=int, default=32)

    parser.add_argument(
        "--procedural-dataset", choices=["alpha", "single"], default=None)
    parser.add_argument("--caide", default=False, action="store_true")
    subparsers = parser.add_subparsers(title="models", dest="model")
    subparsers.required = True

    parser.set_defaults(metric_mode="max")

    # Code LSTM
    lstm = subparsers.add_parser("lstm")
    lstm_subparsers = lstm.add_subparsers(title="losses", dest="loss")
    lstm_subparsers.required = True

    lstm_triplet = lstm_subparsers.add_parser("triplet")

    lstm.add_argument("--char-embedding-size", type=int, default=70)
    lstm.add_argument("--embedding-size", type=int, default=128)
    lstm.add_argument("--char-capacity", nargs="+", type=int, default=[64])
    lstm.add_argument("--line-capacity", nargs="+", type=int, default=[64])
    lstm.add_argument("--dropout-char", type=float, default=0.0)
    lstm.add_argument("--dropout-line", type=float, default=0.0)
    lstm.add_argument("--dropout-fc", type=float, default=0.0)
    lstm.add_argument("--dropout-inter", type=float, default=0.0)

    lstm.add_argument("--max-chars", type=int, default=80)
    lstm.add_argument("--max-lines", type=int, default=100)

    lstm_triplet.add_argument("--margin", required=True, type=float)
    lstm_triplet.add_argument("--classes-per-batch", type=int, default=12)
    lstm_triplet.add_argument("--samples-per-class", type=int, default=6)
    lstm_triplet.add_argument("--extra-samples", type=int, default=0)
    lstm_triplet.set_defaults(metric_mode="min")

    return parser.parse_args()


def setup_optimizer_fn(args):
    if args.optimizer == "adam":
        return lambda p: torch.optim.Adam(p, lr=args.lr)
    elif args.optimizer == "rmsprop":
        return lambda p: torch.optim.RMSprop(p, lr=args.lr)
    raise NotImplementedError("optimizer not implemented")


def get_triplet_lstm_module(args):
    optimizer_fn = setup_optimizer_fn(args)
    net = lstm.TripletLSTM(alphabet=len(nn.ALPHABET) + 1,
                           alphabet_embedding=args.char_embedding_size,
                           char_hidden_size=args.char_capacity[0],
                           line_hidden_size=args.line_capacity[0],
                           margin=args.margin,
                           optimizer_fn=optimizer_fn)
    return net


def get_roc_steps(args):
    return np.linspace(0.0, 2.0, args.threshold_granularity)


if __name__ == "__main__":
    args = argparsing()

    training_sources, validation_sources = nn.load_dataset(args)
    random.shuffle(training_sources)

    input_size = (args.max_lines, args.max_chars)
    extract_fn = extract_hierarchical_features

    training_generator = nn.CodeForTripletGenerator(
        training_sources,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        extra_negatives=args.extra_samples,
        input_size=input_size,
        fn=extract_fn,
        np_cast=False)

    validation_labels = extract_labels([validation_sources])[0]
    validation_sequence = nn.CategoricalCodeSequence(
        validation_sources, validation_labels, input_size=input_size, 
        batch_size=args.validation_batch_size, fn=extract_fn)

    net = get_triplet_lstm_module(args)

    def validate():
        scorer = CompletePairContrastiveScorer(get_roc_steps(args))
        for i in range(len(validation_sequence)):
            X, y = validation_sequence[i]
            y_pred = net.predict_on_batch(X)

            scorer.handle(y, y_pred)

        return scorer.result("eer")

    training_iterable = training_generator()
    for epoch in range(args.max_epochs):
        steps = 0

        loss_aggregator = np.array(0.0)
        n_aggregator = 0
        while steps < args.eval_every:
            X, y = next(training_iterable)
            y = np.array(y)

            _, loss = net.train_on_batch(X, y)
            loss_aggregator += loss * len(y)
            n_aggregator += len(y)
            steps += 1

        print("Finished epoch {}".format(epoch))
        print("Loss: {}".format(loss_aggregator / n_aggregator))
        print("EER: {}".format(validate()))
