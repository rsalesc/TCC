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
from sklearn.preprocessing import LabelEncoder

from scpd.utils import ObjectPairing
from scpd.datasets import CodeforcesDatasetBuilder
from scpd.tf.keras.char import SimilarityCharCNN, TripletCharCNN
from basics import TRAINING_DAT, TEST_DAT, MAGICAL_SEED


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

    def __len__(self):
        batch_size = self._classes_per_batch * self._samples_per_class
        return (len(self._sequence) + batch_size - 1) // batch_size

    def __call__(self):
        while True:
            batch_x, batch_y = zip(
                *self._pick_from_classes(self._classes_per_batch,
                                         self._samples_per_class))
            yield extract_batch_x(batch_x, self._input_size), np.array(batch_y)

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
        batch_y.append(1 if a.author() != b.author() else 0)

    return batch_x, np.array(batch_y).reshape((-1, 1))


def make_pairs(dataset, k1, k2):
    # dataset must be sorted by author
    random.seed(MAGICAL_SEED * 42)
    pairing = ObjectPairing()
    pairs = pairing.make_pairs(dataset, k1=k1, k2=k2)
    return pairs


def load_dataset():
    random.seed(MAGICAL_SEED)

    builder = CodeforcesDatasetBuilder(
        training_size=None,
        test_size=None,
        training_file=TRAINING_DAT,
        test_file=TEST_DAT,
        submissions_per_user=None,
        download=False)

    training_sources, test_sources = builder.extract_raw()
    return training_sources, test_sources


def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy",
        nargs="?",
        default="run",
        choices=["contrastive", "triplet"])
    parser.add_argument("--epoch", default=0, type=int)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--threads", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = argparsing()
    configure(args)

    INPUT_SIZE = 768
    BATCH_SIZE = 32
    CHECKPOINT = ".cache/keras/{strategy}.{name}.{{epoch:02d}}.h5".format(
        strategy=args.strategy, name=args.name)
    to_load = CHECKPOINT.format(epoch=args.epoch)
    initial_epoch = args.epoch

    tb = TensorBoard(
        log_dir="/opt/tensorboard/{}/{}".format(args.strategy, args.name))
    cp = ModelCheckpoint(CHECKPOINT, period=3)
    os.makedirs(".cache/keras", exist_ok=True)
    training_sources, test_sources = load_dataset()

    if args.strategy == "contrastive":
        training_pairs = make_pairs(training_sources, k1=10000, k2=10000)
        random.shuffle(training_pairs)
        training_sequence = CodePairSequence(
            training_pairs, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

        test_pairs = make_pairs(test_sources, k1=1000, k2=1000)
        random.shuffle(test_pairs)
        test_sequence = CodePairSequence(
            test_pairs, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

        nn = SimilarityCharCNN(
            INPUT_SIZE,
            len(ALPHABET) + 1,
            embedding_size=70,
            output_size=20,
            dropout_conv=0.1,
            dropout_fc=0.5)

        if os.path.isfile(to_load):
            print("LOADING PRELOADED MODEL EPOCH={}".format(initial_epoch))
            nn.model = load_model(to_load, nn.loader_objects())
        else:
            nn.build()
            initial_epoch = 0

        nn.compile(base_lr=0.01)

        print(nn.model.summary())
        nn.train(
            training_sequence,
            validation_data=test_sequence,
            callbacks=[tb, cp],
            epochs=1000,
            initial_epoch=initial_epoch)
    elif args.strategy == "triplet":
        random.shuffle(training_sources)
        random.shuffle(test_sources)

        training_generator = CodeForTripletGenerator(
            training_sources,
            classes_per_batch=8,
            samples_per_class=6,
            input_size=INPUT_SIZE)
        test_sequence = CodeSequence(
            test_sources, batch_size=16, input_size=INPUT_SIZE)

        nn = TripletCharCNN(
            INPUT_SIZE,
            len(ALPHABET) + 1,
            embedding_size=70,
            output_size=20,
            dropout_conv=0.1,
            dropout_fc=0.5)

        if os.path.isfile(to_load):
            print("LOADING PRELOADED MODEL EPOCH={}".format(initial_epoch))
            nn.model = load_model(to_load, nn.loader_objects())
        else:
            nn.build()
            initial_epoch = 0

        nn.compile(base_lr=0.01)
        print(nn.model.summary())

        nn.train(
            training_generator(),
            validation_data=test_sequence,
            callbacks=[tb, cp],
            epochs=200,
            steps_per_epoch=len(training_generator),
            initial_epoch=initial_epoch)
    else:
        raise NotImplementedError()
