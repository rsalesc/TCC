import argparse
import tensorflow as tf
import numpy as np
import random
import string
import os
from bisect import bisect
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

from scpd.utils import ObjectPairing
from scpd.datasets import CodeforcesDatasetBuilder
from scpd.tf.keras.char import SimilarityCharCNN, TripletCharCNN
from scpd.tf.keras.metrics import (TripletValidationMetric,
                                   ContrastiveValidationMetric,
                                   FlatPairValidationMetric)
from scpd.tf.keras.callbacks import OfflineMetrics
from constants import TRAINING_DAT, TEST_DAT, MAGICAL_SEED


def configure(args):
    config = tf.ConfigProto()
    if args.threads is not None:
        print("Using {} threads for inter and intra ops.".format(args.threads))
        config.inter_op_parallelism_threads = args.threads
    K.set_session(tf.Session(config=config))


class CodePairSequence(Sequence):
    def __init__(self, pairs, batch_size, input_size=None):
        self._pairs = pairs
        self._batch_size = batch_size
        self._input_size = input_size

    def __len__(self):
        return (len(self._pairs) + self._batch_size - 1) // self._batch_size

    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) *
                            self._batch_size]
        return extract_pair_batch_features(batch, self._input_size)


class FlatCodePairSequence(CodePairSequence):
    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) *
                            self._batch_size]
        return extract_flat_pair_batch_features(batch, self._input_size)


class CodeSequence(Sequence):
    def __init__(self, sequence, batch_size, input_size=None):
        self._sequence = sequence
        self._batch_size = batch_size
        self._input_size = input_size

    def __len__(self):
        return (len(self._sequence) + self._batch_size - 1) // self._batch_size

    def __getitem__(self, idx):
        batch = self._sequence[idx * self._batch_size:(idx + 1) *
                               self._batch_size]
        labels = LabelEncoder().fit_transform(
            list(map(lambda x: x.author(), batch)))
        return extract_batch_x(batch, self._input_size), np.array(labels)


class CodeForTripletGenerator:
    def __init__(self,
                 sequence,
                 classes_per_batch,
                 samples_per_class,
                 extra_negatives=0,
                 input_size=None):
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

    def __len__(self):
        batch_size = (self._classes_per_batch * self._samples_per_class +
                      self._extra_negatives)
        return (len(self._sequence) + batch_size - 1) // batch_size

    def __call__(self):
        while True:
            batch_x, batch_y = zip(
                *self._pick_from_classes(self._classes_per_batch,
                                         self._samples_per_class))
            np_x = extract_batch_x(batch_x, self._input_size)
            np_y = np.array(batch_y)
            p = np.random.permutation(len(batch_x))
            yield np_x[p], np_y[p]

    def _pick_from_classes(self, n, m):
        res = []

        for i in range(self._pointer, self._pointer + n):
            cur = i % len(self._classes)
            if cur == 0:
                random.shuffle(self._classes)
            res.extend(self._pick_from_class(self._classes[cur], m))

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


ALPHABET = sorted(list(string.printable))


def encode_char(char):
    i = bisect(ALPHABET, char)
    if i > 0 and ALPHABET[i - 1] == char:
        return i
    return len(ALPHABET)


def encode_text(text):
    return list(map(encode_char, text))


def extract_features(source, input_size=None):
    # should be more efficient
    res = encode_text(source.fetch())
    if input_size is not None:
        res = res[-input_size:]
        if len(res) < input_size:
            res.extend([len(ALPHABET)] * (input_size - len(res)))
    return np.array(res)


def extract_batch_x(batch, input_size=None):
    batch_x = [extract_features(source, input_size) for source in batch]

    return np.array(batch_x)


def extract_pair_batch_features(batch, input_size=None):
    # no prefetching since AST is not needed for now
    a, b = zip(*batch)
    batch_x = [
        np.array([extract_features(aa, input_size) for aa in a]),
        np.array([extract_features(bb, input_size) for bb in b])
    ]
    batch_y = []
    for a, b in batch:
        batch_y.append(1 if a.author() == b.author() else 0)

    return batch_x, np.array(batch_y)


def extract_flat_pair_batch_features(batch, input_size=None):
    x, y = extract_pair_batch_features(batch, input_size=input_size)
    x = np.array(x)
    x = np.transpose(x, (1, 0, 2)).reshape((x.shape[1] * 2, -1))
    return x, y


def make_pairs(dataset, k1, k2):
    # dataset must be sorted by author
    random.seed(MAGICAL_SEED * 42)
    pairing = ObjectPairing()
    pairs = pairing.make_pairs(dataset, k1=k1, k2=k2)
    return pairs


def load_dataset(args):
    random.seed(MAGICAL_SEED)

    builder = CodeforcesDatasetBuilder(
        training_size=None,
        test_size=None,
        training_file=args.training_file,
        test_file=args.validation_file,
        submissions_per_user=None,
        download=args.download_dataset)

    training_sources, test_sources = builder.extract_raw()
    return training_sources, test_sources


def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=0, type=int)
    parser.add_argument("--max-epochs", default=1000, type=int)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--period", type=int, default=3)
    parser.add_argument("--lr", default=0.05)
    parser.add_argument("--save-to", default=".cache/keras")
    parser.add_argument("--tensorboard-dir", default="/opt/tensorboard")

    parser.add_argument("--training-file", default=TRAINING_DAT)
    parser.add_argument("--validation-file", default=TEST_DAT)
    parser.add_argument("--download-dataset", action="store_true", default=False)

    parser.add_argument("--validation-batch-size", type=int, default=32)
    subparsers = parser.add_subparsers(title="models", dest="model")
    subparsers.required = True

    # Char CNN
    cnn = subparsers.add_parser("cnn")
    cnn_subparsers = cnn.add_subparsers(title="losses", dest="loss")
    cnn_subparsers.required = True

    cnn_triplet = cnn_subparsers.add_parser("triplet")
    cnn_contrastive = cnn_subparsers.add_parser("contrastive")

    cnn.add_argument("--char-embedding-size", type=int, default=70)
    cnn.add_argument("--embedding-size", type=int, default=20)
    cnn.add_argument("--dropout-conv", type=float, default=0.1)
    cnn.add_argument("--dropout-fc", type=float, default=0.5)
    cnn.add_argument("--input-crop", type=int, default=768)

    cnn_triplet.add_argument("--margin", required=True, type=float)
    cnn_triplet.add_argument("--classes-per-batch", type=int, default=24)
    cnn_triplet.add_argument("--samples-per-class", type=int, default=8)
    cnn_triplet.set_defaults(func=run_triplet_cnn)

    cnn_contrastive.add_argument("--margin", required=True, type=float)
    cnn_contrastive.add_argument("--batch-size", type=int, default=32)
    cnn_contrastive.set_defaults(func=run_contrastive_cnn)

    # Code LSTM
    lstm = subparsers.add_parser("lstm")
    lstm_subparsers = lstm.add_subparsers(title="losses", dest="loss")
    lstm_subparsers.required = True

    lstm_triplet = lstm_subparsers.add_parser("triplet")

    lstm.add_argument("--char-embedding-size", type=int, default=70)
    lstm.add_argument("--embedding-size", type=int, default=20)
    lstm.add_argument("--char-capacity", type=int, default=32)
    lstm.add_argument("--line-capacity", type=int, default=32)

    lstm_triplet.add_argument("--margin", required=True, type=float)

    return parser.parse_args()


def setup_callbacks(args, checkpoint):
    tb_dir = os.path.join(args.tensorboard_dir, args.model, args.loss or "unknown", args.name)
    tb = TensorBoard(log_dir=tb_dir)
    cp = ModelCheckpoint(checkpoint, period=args.period)
    return [tb, cp]


def build_scpd_model(nn, path=None):
    if path is None:
        nn.build()
    else:
        nn.model = load_model(path, nn.loader_objects())


def run_triplet_lstm(args, training_sources, validation_sources, load=None, callbacks=[]):
    pass


def run_contrastive_cnn(args, training_sources, validation_sources, load=None, callbacks=[]):
    pass


def run_triplet_cnn(args, training_sources, validation_sources, load=None, callbacks=[]):
    random.shuffle(training_sources)

    training_generator = CodeForTripletGenerator(
        training_sources,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        input_size=args.input_crop)

    validation_pairs = make_pairs(validation_sources, k1=1000, k2=1000)
    random.shuffle(validation_pairs)

    validation_sequence = FlatCodePairSequence(
            validation_pairs, batch_size=args.validation_batch_size, input_size=args.input_crop)

    optimizer = Adam(lr=args.lr)
    nn = TripletCharCNN(
            args.input_crop,
            len(ALPHABET) + 1,
            embedding_size=args.char_embedding_size,
            output_size=args.embedding_size,
            dropout_conv=args.dropout_conv,
            dropout_fc=args.dropout_fc,
            margin=args.margin,
            optimizer=optimizer,
            metric="precision")

    build_scpd_model(nn)
    nn.compile()
    print(nn.model.summary())

    val_threshold_metric = FlatPairValidationMetric(
        np.linspace(0.0, 2.0, 40),
        id="thresholded",
        metric=["precision", "accuracy"],
        argmax=["accuracy"])
    val_margin_metric = FlatPairValidationMetric(
        args.margin, id="margin", metric=["f1", "precision", "accuracy"])
    om = OfflineMetrics(
        on_epoch=[val_threshold_metric, val_margin_metric],
        validation_data=validation_sequence)

    nn.train(
        training_generator(),
        callbacks=[om] + callbacks,
        epochs=args.max_epochs,
        steps_per_epoch=len(training_generator),
        initial_epoch=args.epoch)


def main(args):
    os.makedirs(args.save_to, exist_ok=True)
    checkpoint = os.path.join(args.save_to, "{model}.{loss}.{name}.{{epoch:04d}}.h5").format(model=args.model, loss=(args.loss or "unknown"), name=args.name)
    to_load = checkpoint.format(epoch=args.epoch)
    callbacks = setup_callbacks(args, checkpoint)
    training_sources, validation_sources = load_dataset(args)

    if not os.path.isfile(to_load):
        if args.epoch > 0:
            raise AssertionError("checkpoint for epoch {} was not found".format(args.epoch))
        to_load = None

    if args.func is None:
        raise NotImplementedError()

    args.func(args, training_sources, validation_sources, load=to_load, callbacks=callbacks)


if __name__ == "__main__":
    args = argparsing()
    configure(args)

    import sys
    main(args)
    sys.exit(0)

    #
    #
    #

    INPUT_SIZE = 768
    BATCH_SIZE = 32
    CHECKPOINT = ".cache/keras/{strategy}.{name}.{{epoch:04d}}.h5".format(
        strategy=args.strategy, name=args.name)
    to_load = CHECKPOINT.format(epoch=args.epoch)
    initial_epoch = args.epoch

    tb = TensorBoard(
        log_dir="/opt/tensorboard/{}/{}".format(args.strategy, args.name))
    cp = ModelCheckpoint(
        CHECKPOINT, period=args.period, save_weights_only=False)
    os.makedirs(".cache/keras", exist_ok=True)
    training_sources, test_sources = load_dataset(args)

    test_pairs = make_pairs(test_sources, k1=1000, k2=1000)
    random.shuffle(test_pairs)

    if args.strategy == "contrastive":
        training_pairs = make_pairs(training_sources, k1=10000, k2=10000)
        random.shuffle(training_pairs)
        training_sequence = CodePairSequence(
            training_pairs, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

        test_sequence = CodePairSequence(
            test_pairs, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

        optimizer = Adam(lr=0.01)

        nn = SimilarityCharCNN(
            INPUT_SIZE,
            len(ALPHABET) + 1,
            embedding_size=70,
            output_size=20,
            dropout_conv=0.1,
            dropout_fc=0.5,
            optimizer=optimizer,
            metric="accuracy")

        if os.path.isfile(to_load):
            print("LOADING PRELOADED MODEL EPOCH={}".format(initial_epoch))
            nn.model = load_model(to_load, nn.loader_objects())
        else:
            nn.build()
            initial_epoch = 0

        nn.compile(base_lr=0.01)

        val_metric = ContrastiveValidationMetric(
            np.linspace(0.0, 2.0, 40),
            metric=["accuracy", "precision"],
            argmax=["accuracy", "precision"])
        om = OfflineMetrics(
            on_epoch=[val_metric], validation_data=test_sequence)

        print(nn.model.summary())
        nn.train(
            training_sequence,
            callbacks=[om, tb, cp],
            epochs=1000,
            initial_epoch=initial_epoch)
    elif args.strategy == "triplet":
        random.shuffle(training_sources)

        training_generator = CodeForTripletGenerator(
            training_sources,
            classes_per_batch=24,
            samples_per_class=8,
            input_size=INPUT_SIZE)

        test_sequence = FlatCodePairSequence(
            test_pairs, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

        margin = 0.2
        optimizer = Adam(lr=0.05)
        nn = TripletCharCNN(
            INPUT_SIZE,
            len(ALPHABET) + 1,
            embedding_size=70,
            output_size=20,
            dropout_conv=0.1,
            dropout_fc=0.5,
            margin=margin,
            optimizer=optimizer,
            metric="precision")

        if os.path.isfile(to_load):
            print("LOADING PRELOADED MODEL EPOCH={}".format(initial_epoch))
            nn.model = load_model(to_load, nn.loader_objects())
        else:
            nn.build()
            initial_epoch = 0

        nn.compile()
        print(nn.model.summary())

        val_threshold_metric = FlatPairValidationMetric(
            np.linspace(0.0, 2.0, 40),
            id="thresholded",
            metric=["precision", "accuracy"],
            argmax=["accuracy"])
        val_margin_metric = FlatPairValidationMetric(
            margin, id="margin", metric=["f1", "precision", "accuracy"])
        om = OfflineMetrics(
            on_epoch=[val_threshold_metric, val_margin_metric],
            validation_data=test_sequence)

        nn.train(
            training_generator(),
            callbacks=[om, tb, cp],
            epochs=1000,
            steps_per_epoch=len(training_generator),
            initial_epoch=initial_epoch)
    else:
        raise NotImplementedError()
