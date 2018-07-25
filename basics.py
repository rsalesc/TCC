import argparse
import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from keras.models import load_model
from keras.utils import Sequence
from keras.callbacks import TensorBoard, ModelCheckpoint

from scpd import source
from scpd.utils import ObjectPairing
from scpd.tf.keras.mlp import SimilarityMLP
from scpd.features import (LabelerPairFeatureExtractor, BaseFeatureExtractor,
                           PrefetchBatchFeatureExtractor,
                           SmartPairFeatureExtractor)
from scpd.metrics import BinaryClassificationMetrics
from scpd.datasets import CodeforcesDatasetBuilder
from scpd.learning.scikit import (RandomForestFitter, XGBoostFitter,
                                  DecisionTreeFitter, SVMFitter)

from constants import TRAINING_DAT, TEST_DAT, MAGICAL_SEED

SUBMISSION_API_COUNT = 10000

LEARNING_METHODS = ["xgb", "random", "decision", "svm"]
TRAINING_PKL = "training.pkl.gz"
TEST_PKL = "test.pkl.gz"
AUTHOR_COL = "author"


def extract_features(sources):
    base_extractor = BaseFeatureExtractor()
    prefetch_extractor = PrefetchBatchFeatureExtractor(
        base_extractor, monitor=True)
    return prefetch_extractor.extract(sources)


class RowPairing(ObjectPairing):
    def get_class(self, obj):
        return obj[-1]


def make_pairs(features, k1, k2):
    pairing = RowPairing()
    pairs = pairing.make_pairs(features.tolist(), k1=k1, k2=k2)
    concat_pairs = []
    labels = []
    for pair in pairs:
        concat_pair = []
        concat_pair.append(pair[0][:-1])
        concat_pair.append(pair[1][:-1])
        labels.append(1 if pair[0][-1] != pair[1][-1] else 0)
        concat_pairs.append(concat_pair)
    return np.array(concat_pairs), np.array(labels)


def apply_preprocessing_all(args, training_features, test_features):
    training_authors = training_features[AUTHOR_COL]
    test_authors = test_features[AUTHOR_COL]
    training_features.drop(AUTHOR_COL, axis=1, inplace=True)
    test_features.drop(AUTHOR_COL, axis=1, inplace=True)

    # turn dataframe into ndarray
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features)
    test_features = scaler.transform(test_features)

    # categorical to numerical
    training_authors = LabelEncoder().fit_transform(training_authors)
    test_authors = LabelEncoder().fit_transform(test_authors)

    if args.pca is not None:
        comps = min(args.pca, training_features.shape[1])
        pca = PCA(n_components=comps, random_state=MAGICAL_SEED * 42)
        training_features = pca.fit_transform(training_features)
        test_features = pca.transform(test_features)

        if args.verbose:
            print("PCA explanation: {}".format(pca.explained_variance_ratio_))

    if args.select is not None:
        folder = StratifiedKFold(n_splits=7, random_state=MAGICAL_SEED, shuffle=True)

        clf = RandomForestFitter(
            n_estimators=args.select, folder=folder, random_state=MAGICAL_SEED)
        clf.fit(training_features, df_y=training_authors)
        sel = clf.selector()
        if args.verbose:
            print("Fit selector...")
        training_features = sel.transform(training_features)
        test_features = sel.transform(test_features)
        print(training_features.shape)

    return training_features, training_authors, test_features, test_authors


def apply_preprocessing(args, training_features, test_features):
    training_features, training_authors, test_features, test_authors = (
        apply_preprocessing_all(args, training_features, test_features))
    training_features = np.c_[training_features, training_authors]
    test_features = np.c_[test_features, test_authors]

    training_data, training_labels = make_pairs(
        training_features, k1=10000, k2=10000)
    test_data, test_labels = make_pairs(test_features, k1=1000, k2=1000)

    if args.verbose:
        print("Training features shape: {}".format(training_data.shape))
        print("Test features shape: {}".format(test_data.shape))

    return training_data, training_labels, test_data, test_labels


def apply_preprocessing_for_triplet_mlp(args, training_features,
                                        test_features):
    training_features, training_authors, test_features, test_authors = (
        apply_preprocessing_all(args, training_features, test_features))

    if args.verbose:
        print("Training features shape: {}".format(training_features.shape))
        print("Test features shape: {}".format(test_features.shape))

    return training_features, training_authors, test_features, test_authors


def get_label_distribution(labels):
    neg_labels = 1 - labels
    return np.stack((neg_labels, labels), axis=1)


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
        return [batch_x[:, 0, :], batch_x[:, 1, :]], batch_y


def do_nn(args):
    random.seed(MAGICAL_SEED)
    BATCH_SIZE = 32
    CHECKPOINT = ".cache/keras/mlp.{epoch:04d}.h5"
    LAST_EPOCH = 0

    training_features = pd.read_pickle(TRAINING_PKL, compression="infer")
    test_features = pd.read_pickle(TEST_PKL, compression="infer")
    (training_features, training_labels, test_features,
     test_labels) = apply_preprocessing(args, training_features, test_features)

    training_sequence = RowSequence(
        training_features, training_labels, batch_size=BATCH_SIZE)
    test_sequence = RowSequence(
        test_features, test_labels, batch_size=BATCH_SIZE)
    input_size = training_features.shape[2]

    os.makedirs(".cache/keras", exist_ok=True)
    tb = TensorBoard(log_dir="/opt/tensorboard")
    cp = ModelCheckpoint(CHECKPOINT)

    nn = SimilarityMLP(input_size, 40, 8)

    to_load = CHECKPOINT.format(epoch=LAST_EPOCH)
    initial_epoch = LAST_EPOCH
    if os.path.isfile(to_load):
        print("LOADING PRELOADED MODEL EPOCH={}".format(initial_epoch))
        nn.model = load_model(to_load, nn.loader_objects())
    else:
        nn.build()
        initial_epoch = 0

    nn.compile(0.00005)
    print(nn.model.summary())

    nn.train(
        training_sequence,
        validation_data=test_sequence,
        callbacks=[tb, cp],
        epochs=100,
        initial_epoch=initial_epoch)

    # training_prob_labels = get_label_distribution(training_labels)
    # test_prob_labels = get_label_distribution(test_labels)

    # mlp.execute(
    # args,
    # training_features=training_features,
    # test_features=test_features,
    # training_labels=training_prob_labels,
    # test_labels=test_prob_labels)


def run_main(args):
    random.seed(MAGICAL_SEED)
    method = args.method

    training_features = pd.read_pickle(TRAINING_PKL, compression="infer")
    test_features = pd.read_pickle(TEST_PKL, compression="infer")
    training_features, training_labels, test_features, test_labels = (
        apply_preprocessing(args, training_features, test_features))

    folder = StratifiedKFold(
        n_splits=7, random_state=MAGICAL_SEED, shuffle=True)

    # reshape sets for them to be usable by sklearn fitters
    training_features = training_features.reshape((training_features.shape[0],
                                                   -1))
    test_features = test_features.reshape((test_features.shape[0], -1))

    # expose predictions and labels
    test_pred = None

    if method == "xgb":
        xgb = XGBoostFitter(
            n_estimators=300,
            learning_rate=0.1,
            early_stopping_rounds=50,
            random_state=MAGICAL_SEED,
            folder=folder)
        xgb.fit(training_features, df_y=training_labels)
        test_pred, _ = xgb.predict(test_features)
    elif method == "random":
        random_forest = RandomForestFitter(
            n_estimators=769, folder=folder, random_state=MAGICAL_SEED)
        random_forest.fit(training_features, df_y=training_labels)
        test_pred, _ = random_forest.predict(test_features)
    elif method == "decision":
        clf = DecisionTreeFitter(
            criterion="entropy", folder=folder, random_state=MAGICAL_SEED)
        clf.fit(training_features, df_y=training_labels)
        test_pred, _ = clf.predict(test_features)
    elif method == "svm":
        clf = SVMFitter(dual=False, folder=folder, random_state=MAGICAL_SEED)
        clf.fit(training_features, df_y=training_labels)
        test_pred, _ = clf.predict(test_features)

    else:
        raise AssertionError(
            "Unsupported `{}` learning method.".format(method))

    metrics = BinaryClassificationMetrics(test_pred, test_labels)
    print(metrics)


def build_main_dataset(args):
    random.seed(MAGICAL_SEED)

    builder = CodeforcesDatasetBuilder(
        training_size=10,
        test_size=5,
        training_file=TRAINING_DAT,
        test_file=TEST_DAT,
        submissions_per_user=10,
        download=True)
    training_sources, test_sources = builder.extract()

    random.seed(MAGICAL_SEED * 2)

    training_features = extract_features(training_sources)
    test_features = extract_features(test_sources)

    training_features.to_pickle(TRAINING_PKL, compression="infer")
    test_features.to_pickle(TEST_PKL, compression="infer")

    print(test_features.head())
    print(test_features.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", nargs="?", default="run", choices=["run", "build", "nn"])
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--method", choices=LEARNING_METHODS)
    parser.add_argument("--pca", type=int)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--load", default=False, action="store_true")

    args = parser.parse_args()

    if args.action == "run":
        run_main(args)
    elif args.action == "build":
        build_main_dataset(args)
    elif args.action == "nn":
        do_nn(args)
    else:
        raise AssertionError("Unsupported action argument.")
