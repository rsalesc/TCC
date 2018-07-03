import tensorflow as tf
import numpy as np
import random
import string
import os
from bisect import bisect
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

from scpd.utils import ObjectPairing
from scpd.datasets import CodeforcesDatasetBuilder
from scpd.tf.keras.char import SimilarityCharCNN
from basics import TRAINING_DAT, TEST_DAT, MAGICAL_SEED


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
        return extract_batch_features(batch, self._input_size)


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


def extract_batch_features(batch, input_size=None):
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

    training_sources, test_sources = builder.extract()
    return training_sources, test_sources


if __name__ == "__main__":
    INPUT_SIZE = 768
    BATCH_SIZE = 32
    CHECKPOINT = ".cache/keras/siamesis.{epoch:02d}.h5"
    LAST_EPOCH = 0

    training_sources, test_sources = load_dataset()

    training_pairs = make_pairs(training_sources, k1=10000, k2=10000)
    test_pairs = make_pairs(test_sources, k1=1000, k2=1000)
    random.shuffle(training_pairs)
    random.shuffle(test_pairs)

    training_sequence = CodePairSequence(
        training_pairs, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)
    test_sequence = CodePairSequence(
        test_pairs, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

    os.makedirs(".cache/keras", exist_ok=True)
    tb = TensorBoard(log_dir="/tmp/tensorboard")
    cp = ModelCheckpoint(CHECKPOINT)

    nn = SimilarityCharCNN(
        INPUT_SIZE,
        len(ALPHABET) + 1,
        embedding_size=70,
        output_size=128,
        dropout_conv=0.0,
        dropout_fc=0.4)

    to_load = CHECKPOINT.format(epoch=LAST_EPOCH)
    initial_epoch = LAST_EPOCH + 1
    if os.path.isfile(to_load):
        print("LOADING PRELOADED MODEL EPOCH={}".format(initial_epoch))
        nn.model = load_model(to_load, SimilarityCharCNN.loader_objects())
    else:
        nn.build()
        initial_epoch = 1

    nn.compile()

    print(nn.model.summary())
    nn.train(
        training_sequence,
        validation_data=test_sequence,
        callbacks=[tb, cp],
        epochs=100,
        initial_epoch=initial_epoch)
