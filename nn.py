import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import string
import shutil
import os
import math
import pickle
import time
import keras

from bisect import bisect
from keras import backend as K
from keras.models import load_model
from keras.utils import Sequence
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import LabelEncoder

from scpd.source import SourceCode
from scpd.utils import ObjectPairing, opens, LinearDecay, extract_labels
from scpd.tf.keras.char import SimilarityCharCNN, TripletCharCNN
from scpd.tf.keras.lstm import TripletLineLSTM, SoftmaxLineLSTM
from scpd.tf.keras.mlp import TripletMLP
from scpd.tf.keras.metrics import (TripletValidationMetric,
                                   ContrastiveValidationMetric,
                                   CompletePairValidationMetric,
                                   CategoricalValidationMetric,
                                   FlatPairValidationMetric)
from scpd.tf.keras.callbacks import OfflineMetrics, SaverCheckpoint
from basics import (TRAINING_PKL, TEST_PKL,
                    apply_preprocessing_for_triplet_mlp, RowPairing)
import dataset
from constants import TRAINING_DAT, TEST_DAT, MAGICAL_SEED


def configure(args):
    config = tf.ConfigProto()
    if args.threads is not None:
        print("Using {} threads for inter and intra ops.".format(args.threads))
        config.inter_op_parallelism_threads = args.threads
    K.set_session(tf.Session(config=config))


def force_rmtree(root_dir):
    '''
    rmtree doesn't work when no write bit in linux or read-only in windows
    force_rmtree recursively walk, do chmod and then remove
    '''
    import stat
    from os import path, rmdir, remove, chmod, walk
    for root, dirs, files in walk(root_dir, topdown=False):
        for name in files:
            file_path = path.join(root, name)
            chmod(file_path, stat.S_IWUSR)
            remove(file_path)
        for name in dirs:
            dir_path = path.join(root, name)
            chmod(dir_path, stat.S_IWUSR)
            rmdir(dir_path)
    rmdir(root_dir)


def build_alpha(alpha, classes, samples, length):
    source = string.ascii_lowercase[:6]
    assert alpha <= 6

    sources = []
    for i in range(classes):
        candidates = np.random.randint(0, 6, alpha)
        for j in range(samples):
            chars = []
            for k in range(random.randint(1, length)):
                chars.append(source[np.random.choice(candidates)])
            code = "".join(chars)
            sources.append(SourceCode("guy_{}".format(i), code))

    return sources


def build_single_alpha(samples, length):
    sources = []
    for c in string.ascii_lowercase:
        for j in range(samples):
            code = "".join([c] * random.randint(1, length))
            sources.append(SourceCode("guy_{}".format(c), code))
    return sources


class PerIterationTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iterations = 0

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)

        logs = logs or {}
        self._iterations += 1
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name + "_iterations"
            self.writer.add_summary(summary, self._iterations)
        self.writer.flush()


class RowSequence(Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._size = x.shape[0]
        if shuffle:
            p = np.random.permutation(x.shape[0])
            self._x = x[p]
            self._y = y[p]

    def __len__(self):
        return (self._size + self._batch_size - 1) // self._batch_size

    def __getitem__(self, idx):
        batch_x = self._x[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]
        return np.array(batch_x), np.array(batch_y)


class CodePairSequence(Sequence):
    def __init__(self,
                 pairs,
                 batch_size,
                 input_size=None,
                 fn=None,
                 fn_author=None):
        assert fn is not None
        self._pairs = pairs
        self._batch_size = batch_size
        self._input_size = input_size
        self._ex = NeuralFeatureExtractor(fn, input_size, fn_author=fn_author)

    def __len__(self):
        return (len(self._pairs) + self._batch_size - 1) // self._batch_size

    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) *
                            self._batch_size]
        return self._ex.extract_pair_batch_features(batch)


class FlatCodePairSequence(CodePairSequence):
    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) *
                            self._batch_size]
        return self._ex.extract_flat_pair_batch_features(batch)


class CodeSequence(Sequence):
    def __init__(self, sequence, batch_size, input_size=None, fn=None):
        assert fn is not None
        self._sequence = sequence
        self._batch_size = batch_size
        self._input_size = input_size
        self._ex = NeuralFeatureExtractor(fn, input_size)

    def __len__(self):
        return (len(self._sequence) + self._batch_size - 1) // self._batch_size

    def __getitem__(self, idx):
        batch = self._sequence[idx * self._batch_size:(idx + 1) *
                               self._batch_size]
        labels = LabelEncoder().fit_transform(
            list(map(lambda x: x.author(), batch)))
        return self._ex.extract_batch_x(batch), np.array(labels)


class CategoricalCodeSequence(Sequence):
    def __init__(self, sequence, labels, batch_size, input_size=None, fn=None):
        assert fn is not None
        assert len(sequence) == len(labels)
        self._sequence = sequence
        self._labels = labels
        self._batch_size = batch_size
        self._input_size = input_size
        self._ex = NeuralFeatureExtractor(fn, input_size)

    def __len__(self):
        return (len(self._sequence) + self._batch_size - 1) // self._batch_size

    def __getitem__(self, idx):
        batch = self._sequence[idx * self._batch_size:(idx + 1) *
                               self._batch_size]
        labels = self._labels[idx * self._batch_size:(idx + 1) *
                              self._batch_size]
        return self._ex.extract_batch_x(batch), np.array(labels)


class CodeForTripletGenerator:
    def __init__(self,
                 sequence,
                 classes_per_batch,
                 samples_per_class,
                 extra_negatives=0,
                 input_size=None,
                 fn=None,
                 np_cast=True):
        assert fn is not None
        self._sequence = sequence.copy()
        self._labels = self._generate_labels()
        self._classes = self._generate_classes()
        self._indices_per_class = self._generate_indices_per_class()
        self._pointers_per_class = self._generate_pointers_per_class()
        self._pointer = 0
        self._classes_per_batch = classes_per_batch
        self._samples_per_class = samples_per_class
        self._input_size = input_size
        self._extra_negatives = extra_negatives
        self._ex = NeuralFeatureExtractor(fn, input_size)
        self._np_cast = np_cast

    def __len__(self):
        batch_size = (self._classes_per_batch * self._samples_per_class +
                      self._extra_negatives)
        return (len(self._sequence) + batch_size - 1) // batch_size

    def __call__(self):
        while True:
            batch_x, batch_y = zip(
                *self._pick_from_classes(self._classes_per_batch,
                                         self._samples_per_class,
                                         self._extra_negatives))

            batch_x = self._ex.extract_batch_x(batch_x)
            p = np.random.permutation(len(batch_x))
            if self._np_cast:
                np_x = np.array(batch_x)
                np_y = np.array(batch_y)
                yield np_x[p], np_y[p]
            else:
                X = [batch_x[i] for i in p]
                y = [batch_y[i] for i in p]
                yield X, y

    def _pick_from_classes(self, n, m, neg):
        res = []

        for i in range(self._pointer, self._pointer + n):
            cur = i % len(self._classes)
            if cur == 0:
                random.shuffle(self._classes)
            res.extend(self._pick_from_class(self._classes[cur], m))

        res.extend(self._pick_random_samples(neg))

        self._pointer = (self._pointer + n) % len(self._classes)
        return res

    def _pick_from_class(self, label, n):
        if n == 0:
            return []

        indices = self._indices_per_class[label]
        if self._pointers_per_class[label] == 0:
            random.shuffle(indices)

        til = min(self._pointers_per_class[label] + n, len(indices))
        chosen_indices = indices[self._pointers_per_class[label]:til]

        self._pointers_per_class[label] = 0 if len(indices) == til else til
        x = [self._sequence[i] for i in chosen_indices]
        y = [self._labels[i] for i in chosen_indices]
        return list(zip(*(x, y))) + self._pick_from_class(
            label, n - len(chosen_indices))

    def _pick_random_samples(self, n):
        p = np.random.permutation(len(self._sequence))[:n]
        x = [self._sequence[i] for i in p]
        y = [self._labels[i] for i in p]
        return list(zip(*(x, y)))

    def _generate_classes(self):
        return list(set(self._labels))

    def _generate_indices_per_class(self):
        res = {}
        for i, label in enumerate(self._labels):
            if label not in res:
                res[label] = []
            res[label].append(i)
        return res

    def _generate_pointers_per_class(self):
        res = {}
        for label in self._labels:
            res[label] = 0
        return res

    def _generate_labels(self):
        return LabelEncoder().fit_transform(
            list(map(lambda x: x.author(), self._sequence)))


class DataframeForTripletGenerator(CodeForTripletGenerator):
    def __init__(self, x, y, *args, **kwargs):
        if "fn" not in kwargs:

            def fn(x, _):
                return x[0]

            kwargs["fn"] = fn
        super().__init__(list(zip(*(x.tolist(), y.tolist()))), *args, **kwargs)

    def _generate_labels(self):
        features, labels = zip(*self._sequence)
        return labels


ALPHABET = sorted(list(string.printable))


def encode_char(char):
    i = bisect(ALPHABET, char)
    if i > 0 and ALPHABET[i - 1] == char:
        return i
    return 0


def encode_text(text):
    return list(map(encode_char, text))


def crop_or_extend(s, crop_size, pad=0):
    g = s
    if crop_size > 0:
        g = s[:crop_size]
    else:
        g = s[crop_size:]
    if len(g) < abs(crop_size):
        g.extend([pad] * (abs(crop_size) - len(g)))
    return g


def window_or_extend(s, window_size, pad=0):
    if abs(window_size) > len(s):
        return crop_or_extend(s, window_size, pad=pad)
    window_size = abs(window_size)
    st = np.random.randint(low=0, high=len(s) - window_size)
    return s[st:st + window_size]


def extract_cnn_features(source, input_size=None):
    # should be more efficient
    res = encode_text(source.fetch())
    if input_size is not None:
        res = crop_or_extend(res, -input_size)
    return np.array(res)


def extract_hierarchical_features(source, input_size=None):
    assert isinstance(input_size, tuple) and len(input_size) == 2
    max_lines, max_chars = input_size
    assert max_lines is not None
    assert max_chars is not None
    lines = [crop_or_extend(encode_text(x), max_chars)
             for x in source.fetch().splitlines()]
    lines = crop_or_extend(lines, -max_lines, pad=[0] * max_chars)
    return np.array(lines)


def _extract_window_hierarchical_features(source, input_size=None, dropout=0.0):
    assert isinstance(input_size, tuple) and len(input_size) == 2
    max_lines, max_chars = input_size
    assert max_lines is not None
    assert max_chars is not None
    keep_prob = 1. - dropout
    lines = [crop_or_extend(encode_text(x), max_chars)
             for x in source.fetch().splitlines()]
    lines = window_or_extend(lines, -max_lines, pad=[0] * max_chars)
    lines = [line for line in lines if np.random.rand() < keep_prob]
    print(np.array(lines).shape)
    return np.array(lines)


def HierarchicalWindowFeatureExtractor(dropout):
    def fn(source, input_size=None):
        return _extract_window_hierarchical_features(source,
                                                     input_size=input_size,
                                                     dropout=dropout)

    return fn


def fn_author_default(x):
    return x.author()


class NeuralFeatureExtractor:
    def __init__(self, fn, input_size=None, fn_author=None):
        self._fn = fn
        self._fn_author = fn_author or fn_author_default
        self._input_size = input_size

    def extract_batch_x(self, batch):
        batch_x = [self._fn(source, self._input_size) for source in batch]

        return np.array(batch_x)

    def extract_pair_batch_features(self, batch):
        input_size = self._input_size
        # no prefetching since AST is not needed for now
        a, b = zip(*batch)
        batch_x = [
            np.array([self._fn(aa, input_size) for aa in a]),
            np.array([self._fn(bb, input_size) for bb in b])
        ]
        batch_y = []
        for a, b in batch:
            batch_y.append(1
                           if self._fn_author(a) == self._fn_author(b) else 0)

        return batch_x, np.array(batch_y)

    def extract_flat_pair_batch_features(self, batch):
        x, y = self.extract_pair_batch_features(batch)
        x = np.array(x)
        x = x.reshape((x.shape[1] * 2, ) + x.shape[2:])
        return x, y


def make_pairs(dataset, k1, k2, pairing=ObjectPairing()):
    # dataset must be sorted by author
    random.seed(MAGICAL_SEED * 42)
    pairs = pairing.make_pairs(dataset, k1=k1, k2=k2)
    return pairs


def load_embedding_dataset(args):
    random.seed(42 * MAGICAL_SEED)

    if args.procedural_dataset == "alpha":
        return build_alpha(3, 12, 40, args.input_crop)
    if args.procedural_dataset == "single":
        return build_single_alpha(20, args.input_crop)

    return dataset.preloaded([args.validation_file], caide=args.caide)[0]


def load_dataset(args):
    if hasattr(args, "dataset_func") and args.dataset_func:
        return args.dataset_func(args)
    random.seed(MAGICAL_SEED)

    if args.procedural_dataset == "alpha":
        return build_alpha(3, 400, 40, args.input_crop), build_alpha(
            3, 12, 40, args.input_crop)
    if args.procedural_dataset == "single":
        return build_single_alpha(50, args.input_crop), build_single_alpha(
            20, args.input_crop)

    training_sources, validation_sources = dataset.preloaded(
        [args.training_file, args.validation_file], caide=args.caide)
    return training_sources, validation_sources


def load_gcj_easiest_dataset(args):
    random.seed(MAGICAL_SEED)

    training_sources, validation_sources, _ = dataset.preloaded_gcj_easiest(
        [args.training_file], caide=args.caide)
    return training_sources, validation_sources


def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=0, type=int)
    parser.add_argument("--max-epochs", default=1000, type=int)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--lr-decay", default=0, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--optimizer", default="adam",
                        choices=["adam", "rmsprop"])
    parser.add_argument("--save-to", default=".cache/keras")
    parser.add_argument("--no-checkpoint", action="store_true", default=False)
    parser.add_argument("--tensorboard-dir", default="/opt/tensorboard")
    parser.add_argument(
        "--reset-tensorboard", action="store_true", default=False)
    parser.add_argument("--threshold-granularity", type=int, default=256)

    parser.add_argument("--training-file", default=TRAINING_DAT)
    parser.add_argument("--validation-file", default=TEST_DAT)
    parser.add_argument("--embedding-file", default=TEST_DAT)
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

    # MLP
    mlp = subparsers.add_parser("mlp")
    mlp_subparsers = mlp.add_subparsers(title="losses", dest="loss")
    mlp_subparsers.required = True

    mlp_triplet = mlp_subparsers.add_parser("triplet")

    mlp.add_argument("--dropout", type=float, default=0.0)
    mlp.add_argument("--hidden-size", nargs="+", type=int, default=[64])
    mlp.add_argument("--embedding-size", type=int, default=64)
    mlp.add_argument("--pca", type=int, default=None)
    mlp.add_argument("--verbose", default=False, action="store_true")
    mlp.add_argument("--select", type=int, default=None)

    mlp_triplet.add_argument("--margin", required=True, type=float)
    mlp_triplet.add_argument("--classes-per-batch", type=int, default=12)
    mlp_triplet.add_argument("--samples-per-class", type=int, default=8)
    mlp_triplet.set_defaults(func=run_triplet_mlp)
    mlp_triplet.set_defaults(emb_func=get_embedding_triplet_mlp)

    # Char CNN
    cnn = subparsers.add_parser("cnn")
    cnn_subparsers = cnn.add_subparsers(title="losses", dest="loss")
    cnn_subparsers.required = True

    cnn_triplet = cnn_subparsers.add_parser("triplet")
    cnn_contrastive = cnn_subparsers.add_parser("contrastive")

    cnn.add_argument("--char-embedding-size", type=int, default=70)
    cnn.add_argument("--embedding-size", type=int, default=128)
    cnn.add_argument("--dropout-conv", type=float, default=0.0)
    cnn.add_argument("--dropout-fc", type=float, default=0.0)
    cnn.add_argument("--input-crop", type=int, default=768)

    cnn_triplet.add_argument("--margin", required=True, type=float)
    cnn_triplet.add_argument("--classes-per-batch", type=int, default=24)
    cnn_triplet.add_argument("--samples-per-class", type=int, default=8)
    cnn_triplet.set_defaults(func=run_triplet_cnn)
    cnn_triplet.set_defaults(emb_func=get_embedding_triplet_cnn)

    cnn_contrastive.add_argument("--margin", required=True, type=float)
    cnn_contrastive.add_argument("--batch-size", type=int, default=32)
    cnn_contrastive.set_defaults(func=run_contrastive_cnn)

    # Code LSTM
    lstm = subparsers.add_parser("lstm")
    lstm_subparsers = lstm.add_subparsers(title="losses", dest="loss")
    lstm_subparsers.required = True

    lstm_triplet = lstm_subparsers.add_parser("triplet")
    lstm_softmax = lstm_subparsers.add_parser("softmax")

    lstm.add_argument("--char-embedding-size", type=int, default=70)
    lstm.add_argument("--embedding-size", type=int, default=128)
    lstm.add_argument("--char-capacity", nargs="+", type=int, default=[64])
    lstm.add_argument("--line-capacity", nargs="+", type=int, default=[64])
    lstm.add_argument("--dropout-char", type=float, default=0.0)
    lstm.add_argument("--dropout-line", type=float, default=0.0)
    lstm.add_argument("--dropout-fc", type=float, default=0.0)
    lstm.add_argument("--dropout-inter", type=float, default=0.0)

    lstm.add_argument("--max-chars", type=int, default=80)
    lstm.add_argument("--max-lines", type=int, default=300)

    lstm_triplet.add_argument("--margin", required=True, type=float)
    lstm_triplet.add_argument("--classes-per-batch", type=int, default=12)
    lstm_triplet.add_argument("--samples-per-class", type=int, default=6)
    lstm_triplet.add_argument("--extra-samples", type=int, default=0)
    lstm_triplet.set_defaults(func=run_triplet_lstm)
    lstm_triplet.set_defaults(emb_func=get_embedding_triplet_lstm)
    lstm_triplet.set_defaults(metric_mode="min")

    lstm_softmax.add_argument("--batch-size", type=int, default=32)
    lstm_softmax.add_argument("--hidden-size", nargs="*", type=int, default=[])
    lstm_softmax.add_argument("--classes", type=int, required=True)
    lstm_softmax.add_argument("--pretrained", type=str, default=None)
    lstm_softmax.add_argument("--pretrained-freeze", default=False,
                              action="store_true")
    lstm_softmax.set_defaults(func=run_softmax_lstm)
    lstm_softmax.set_defaults(emb_func=None)
    lstm_softmax.set_defaults(dataset_func=load_gcj_easiest_dataset)
    lstm_softmax.set_defaults(margin=0.2)

    return parser.parse_args()


def setup_tensorboard(args, nn):
    if not os.path.isdir(args.tensorboard_dir):
        raise AssertionError("{} does not exist", args.tensorboard_dir)
    tb_dir = os.path.abspath(
        os.path.join(args.tensorboard_dir, args.model, args.loss or "unknown",
                     args.name))
    if (args.reset_tensorboard or args.epoch == 0) and os.path.isdir(tb_dir):
        print("Resetting TensorBoard logs...")
        shutil.rmtree(tb_dir, ignore_errors=True)
        time.sleep(5)
    params = {"log_dir": tb_dir}
    print("Logging TensorBoard to {}...".format(tb_dir))
    if args.embedding_file is not None and args.embedding_period:
        layer_names = nn.embeddings_to_watch()

        if len(layer_names) > 0 and args.emb_func is not None:
            print("Projecting embeddings...")
            metadata_path = os.path.join(tb_dir, ".embeddings/out.tsv")

            params["embeddings_freq"] = args.embedding_period
            metadata = {}
            for name in layer_names:
                metadata[name] = metadata_path

            params["embeddings_layer_names"] = layer_names
            params["embeddings_metadata"] = os.path.relpath(
                metadata_path, start=tb_dir)

            x, labels, headers = args.emb_func(args)
            params["embeddings_data"] = x

            with opens(metadata_path, "w", encoding="utf-8") as f:
                if headers is not None and len(headers) > 1:
                    f.write("\t".join(headers) + "\n")
                lines = []
                for entry in labels:
                    lines.append("\t".join(entry))

                f.write("\n".join(lines))

    return PerIterationTensorBoard(**params)


def setup_callbacks(args, checkpoint):
    res = []
    if not args.no_checkpoint:
        basename = os.path.splitext(checkpoint)[0]
        args_fn = "{}.{}".format(basename, "args.pkl")
        with opens(args_fn, "wb") as f:
            pickle.dump(args, f)
        if not args.period or args.period == 0:
            res.append(SaverCheckpoint(
                checkpoint,
                save_best_only=True,
                monitor="best_metric",
                verbose=1,
                mode=args.metric_mode))
            print("Saving only BEST MODEL")
        else:
            res.append(SaverCheckpoint(checkpoint, period=args.period))

    res.append(EarlyStopping(
        patience=args.patience,
        monitor="best_metric",
        mode=args.metric_mode,
        min_delta=0.00005))

    print("Early stopping on metric mode: {}".format(args.metric_mode))
    res.append(LearningRateScheduler(LinearDecay(args.lr_decay)))
    return res


def setup_optimizer(args):
    if args.optimizer == "adam":
        return Adam(lr=args.lr)
    elif args.optimizer == "rmsprop":
        return RMSprop(lr=args.lr)
    raise NotImplementedError("optimizer not implemented")


def build_scpd_model(nn, path=None):
    nn.build()

    if path is not None:
        path_tf = "{}.tf".format(path)
        saver = tf.train.Saver()
        sess = keras.backend.get_session()
        saver.restore(sess, path_tf)


def get_embedding_triplet_mlp(args):
    training_features = pd.read_pickle(TRAINING_PKL, compression="infer")
    test_features = pd.read_pickle(TEST_PKL, compression="infer")
    (training_features, training_labels, test_features,
     test_labels) = apply_preprocessing_for_triplet_mlp(
         args, training_features, test_features)

    ex = NeuralFeatureExtractor(fn=lambda x, _: x)
    x = ex.extract_batch_x(test_features.tolist())
    labels = list(map(lambda x: [str(x)], test_labels))
    return x, labels, ["author"]


def run_triplet_mlp(args,
                    training_sources,
                    validation_sources,
                    load=None,
                    callbacks=[]):

    training_features = pd.read_pickle(TRAINING_PKL, compression="infer")
    test_features = pd.read_pickle(TEST_PKL, compression="infer")
    (training_features, training_labels, test_features,
     test_labels) = apply_preprocessing_for_triplet_mlp(
         args, training_features, test_features)

    input_size = training_features.shape[1]

    training_generator = DataframeForTripletGenerator(
        training_features, training_labels, args.classes_per_batch,
        args.samples_per_class)

    test_data = list(zip(*(test_features, test_labels)))
    validation_pairs = make_pairs(
        test_data, k1=1000, k2=1000, pairing=RowPairing())
    random.shuffle(validation_pairs)

    validation_sequence = FlatCodePairSequence(
        validation_pairs,
        batch_size=args.validation_batch_size,
        fn=lambda x, _: x[0],
        fn_author=lambda x: x[1])

    optimizer = setup_optimizer(args)
    nn = TripletMLP(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=args.embedding_size,
        dropout=args.dropout,
        margin=args.margin,
        optimizer=optimizer,
        metric=["precision", "recall", "accuracy"])

    build_scpd_model(nn)
    nn.compile()
    print(nn.model.summary())

    val_threshold_metric = FlatPairValidationMetric(
        np.linspace(0.0, 2.0, args.threshold_granularity),
        id="thresholded",
        metric=["precision", "accuracy", "recall"],
        argmax="accuracy")
    om = OfflineMetrics(
        on_epoch=[val_threshold_metric],
        validation_data=validation_sequence,
        best_metric="val_thresholded_accuracy")
    tb = setup_tensorboard(args, nn)

    nn.train(
        training_generator(),
        callbacks=[om, tb] + callbacks,
        epochs=args.max_epochs,
        steps_per_epoch=args.eval_every or len(training_generator),
        initial_epoch=args.epoch)


def get_embedding_triplet_lstm(args):
    ex = NeuralFeatureExtractor(
        extract_hierarchical_features,
        input_size=(args.max_lines, args.max_chars))
    sources = load_embedding_dataset(args)
    labels = list(map(lambda x: [x.author()], sources))
    x = ex.extract_batch_x(sources)
    return x, labels, ["author"]


def get_triplet_lstm_nn(args, optimizer):
    return TripletLineLSTM(
        len(ALPHABET) + 1,
        embedding_size=args.char_embedding_size,
        output_size=args.embedding_size,
        char_capacity=args.char_capacity,
        line_capacity=args.line_capacity,
        dropout_char=args.dropout_char,
        dropout_line=args.dropout_line,
        dropout_fc=args.dropout_fc,
        margin=args.margin,
        optimizer=optimizer,
        metric=["eer", "accuracy"])


def run_triplet_lstm(args,
                     training_sources,
                     validation_sources,
                     load=None,
                     callbacks=[]):
    random.shuffle(training_sources)
    input_size = (args.max_lines, args.max_chars)
    extract_fn = HierarchicalWindowFeatureExtractor(args.dropout_inter)

    training_generator = CodeForTripletGenerator(
        training_sources,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        extra_negatives=args.extra_samples,
        input_size=input_size,
        fn=extract_fn)

    validation_labels = extract_labels([validation_sources])[0]
    validation_sequence = (
        CategoricalCodeSequence(validation_sources,
                                validation_labels,
                                input_size=input_size,
                                batch_size=args.validation_batch_size,
                                fn=extract_fn))

    optimizer = setup_optimizer(args)
    nn = get_triplet_lstm_nn(args, optimizer)

    build_scpd_model(nn)
    nn.compile()
    print(nn.model.summary())

    val_metric = CompletePairValidationMetric(
        np.linspace(0.0, 2.0, args.threshold_granularity),
        id="complete",
        metric=["eer"])
    om = OfflineMetrics(
        on_epoch=[val_metric],
        validation_data=validation_sequence,
        best_metric="val_complete_eer")
    tb = setup_tensorboard(args, nn)

    nn.train(
        training_generator(),
        callbacks=[om, tb] + callbacks,
        epochs=args.max_epochs,
        steps_per_epoch=args.eval_every or len(training_generator),
        initial_epoch=args.epoch)


def run_contrastive_cnn(args,
                        training_sources,
                        validation_sources,
                        load=None,
                        callbacks=[]):
    pass


def get_embedding_triplet_cnn(args):
    ex = NeuralFeatureExtractor(
        extract_cnn_features, input_size=args.input_crop)
    sources = load_embedding_dataset(args)
    labels = list(map(lambda x: [x.author()], sources))
    x = ex.extract_batch_x(sources)
    return x, labels, ["author"]


def run_triplet_cnn(args,
                    training_sources,
                    validation_sources,
                    load=None,
                    callbacks=[]):
    random.shuffle(training_sources)
    input_size = args.input_crop
    extract_fn = extract_cnn_features

    training_generator = CodeForTripletGenerator(
        training_sources,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        input_size=input_size,
        fn=extract_fn)

    validation_pairs = make_pairs(validation_sources, k1=1000, k2=1000)
    random.shuffle(validation_pairs)

    validation_sequence = FlatCodePairSequence(
        validation_pairs,
        batch_size=args.validation_batch_size,
        input_size=input_size,
        fn=extract_fn)

    optimizer = setup_optimizer(args)
    nn = TripletCharCNN(
        args.input_crop,
        len(ALPHABET) + 1,
        embedding_size=args.char_embedding_size,
        output_size=args.embedding_size,
        dropout_conv=args.dropout_conv,
        dropout_fc=args.dropout_fc,
        margin=args.margin,
        optimizer=optimizer,
        metric=["precision", "recall", "accuracy"])

    build_scpd_model(nn)
    nn.compile()
    print(nn.model.summary())

    val_threshold_metric = FlatPairValidationMetric(
        np.linspace(0.0, 2.0, args.threshold_granularity),
        id="thresholded",
        metric=["precision", "accuracy", "recall"],
        argmax="accuracy")
    om = OfflineMetrics(
        on_epoch=[val_threshold_metric],
        validation_data=validation_sequence,
        best_metric="val_thresholded_accuracy")
    tb = setup_tensorboard(args, nn)

    nn.train(
        training_generator(),
        callbacks=[om, tb] + callbacks,
        epochs=args.max_epochs,
        steps_per_epoch=args.eval_every or len(training_generator),
        initial_epoch=args.epoch)


def get_softmax_lstm_nn(args, optimize):
    oargs = args

    if args.pretrained:
        basename = os.path.splitext(args.pretrained)[0]
        args_fn = "{}.{}".format(basename, "args.pkl")
        with opens(args_fn, "rb") as f:
            args = pickle.load(f)

    def get_pretrain():
        if not oargs.pretrained:
            return None

        nn = get_triplet_lstm_nn(args, None)
        build_scpd_model(nn, path=oargs.pretrained)

        return nn

    pretrain_nn = get_pretrain()

    optimizer = None
    if optimize:
        optimizer = setup_optimizer(oargs)

    return SoftmaxLineLSTM(
        len(ALPHABET) + 1,
        embedding_size=args.char_embedding_size,
        output_size=args.embedding_size,
        char_capacity=args.char_capacity,
        line_capacity=args.line_capacity,
        dropout_char=args.dropout_char,
        dropout_line=args.dropout_line,
        dropout_fc=args.dropout_fc,
        hidden_size=oargs.hidden_size,
        optimizer=optimizer,
        classes=oargs.classes,
        pretrained=pretrain_nn,
        metric=["accuracy"])


def run_softmax_lstm(args,
                     training_sources,
                     validation_sources,
                     load=None,
                     callbacks=[]):
    random.shuffle(training_sources)
    random.shuffle(validation_sources)
    input_size = (args.max_lines, args.max_chars)
    extract_fn = HierarchicalWindowFeatureExtractor(args.dropout_inter)

    training_labels, validation_labels = extract_labels(
        [training_sources, validation_sources],
        args.classes,
        one_hot=True)

    training_sequence = (
        CategoricalCodeSequence(training_sources,
                                training_labels,
                                input_size=input_size,
                                batch_size=args.batch_size,
                                fn=extract_fn))

    validation_sequence = (
        CategoricalCodeSequence(validation_sources,
                                validation_labels,
                                input_size=input_size,
                                batch_size=args.validation_batch_size,
                                fn=extract_fn))

    nn = get_softmax_lstm_nn(args, optimize=True)
    build_scpd_model(nn)
    nn.compile()
    print(nn.model.summary())

    val_metric = CategoricalValidationMetric(
        metric=["accuracy"])
    om = OfflineMetrics(
        on_epoch=[val_metric],
        validation_data=validation_sequence,
        best_metric="val_accuracy")
    tb = setup_tensorboard(args, nn)

    nn.train(
        training_sequence,
        callbacks=[om, tb] + callbacks,
        epochs=args.max_epochs,
        shuffle=True,
        initial_epoch=args.epoch)


def main(args):
    os.makedirs(args.save_to, exist_ok=True)
    checkpoint = os.path.join(args.save_to,
                              "{model}.{name}.h5").format(
                                  model=args.model,
                                  name=args.name)
    to_load = checkpoint
    callbacks = setup_callbacks(args, checkpoint)
    training_sources, validation_sources = load_dataset(args)

    if not os.path.isfile(to_load):
        if args.epoch > 0:
            raise AssertionError(
                "checkpoint for epoch {} was not found".format(args.epoch))
        to_load = None

    if args.func is None:
        raise NotImplementedError()

    args.func(
        args,
        training_sources,
        validation_sources,
        load=to_load,
        callbacks=callbacks)


if __name__ == "__main__":
    args = argparsing()
    configure(args)
    main(args)
