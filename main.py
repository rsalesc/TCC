import argparse
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

from scpd import source
from scpd.features import (LabelerPairFeatureExtractor, BaseFeatureExtractor,
                           PrefetchBatchFeatureExtractor,
                           SmartPairFeatureExtractor)
from scpd.metrics import BinaryClassificationMetrics
from scpd.datasets import CodeforcesDatasetBuilder
from scpd.learning.scikit import RandomForestFitter, XGBoostFitter

MAGICAL_SEED = 42
SUBMISSION_API_COUNT = 10000

LEARNING_METHODS = ["xgb", "random"]
TRAINING_PKL = "training.pkl.gz"
TEST_PKL = "test.pkl.gz"


def extract_features(pairs):
    base_extractor = BaseFeatureExtractor()
    prefetch_extractor = PrefetchBatchFeatureExtractor(
        base_extractor, monitor=True)
    pair_extractor = SmartPairFeatureExtractor(prefetch_extractor)
    labeler = LabelerPairFeatureExtractor(pair_extractor)
    return labeler.extract(pairs)


def apply_preprocessing(args, training_features, test_features):
    training_labels = training_features["label"]
    test_labels = test_features["label"]
    training_features.drop("label", axis=1, inplace=True)
    test_features.drop("label", axis=1, inplace=True)

    # turn dataframe into ndarray
    training_features = preprocessing.scale(training_features)
    test_features = preprocessing.scale(test_features)

    if args.pca is not None:
        comps = min(args.pca, training_features.shape[1])
        pca = PCA(n_components=comps, random_state=MAGICAL_SEED * 42)
        training_features = pca.fit_transform(training_features)
        test_features = pca.transform(test_features)

        print(pca.explained_variance_ratio_)

    return training_features, test_features, training_labels, test_labels


def do_nn(args):
    random.seed(MAGICAL_SEED)
    train = args.train

    training_features = pd.read_pickle(TRAINING_PKL, compression="infer")
    test_features = pd.read_pickle(TEST_PKL, compression="infer")
    (training_features, test_features, training_labels,
     test_labels) = apply_preprocessing(args, training_features, test_features)


def run_main(args):
    random.seed(MAGICAL_SEED)
    method = args.method

    training_features = pd.read_pickle("training.pkl.gz", compression="infer")
    test_features = pd.read_pickle("test.pkl.gz", compression="infer")
    (training_features, test_features, training_labels,
     test_labels) = apply_preprocessing(args, training_features, test_features)

    folder = StratifiedKFold(
        n_splits=7, random_state=MAGICAL_SEED, shuffle=True)

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
            n_estimators=100, folder=folder, random_state=MAGICAL_SEED)
        random_forest.fit(training_features, df_y=training_labels)
        test_pred, _ = random_forest.predict(test_features)
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
        training_file="submissions_training.dat",
        test_file="submissions_test.dat",
        submissions_per_user=10,
        download=False)
    training_sources, test_sources = builder.extract()

    random.seed(MAGICAL_SEED * 2)
    training_pairs = source.make_pairs(training_sources, k1=10000, k2=10000)
    test_pairs = source.make_pairs(test_sources, k1=1000, k2=1000)

    training_features = extract_features(training_pairs)
    test_features = extract_features(test_pairs)

    training_features.to_pickle("training.pkl.gz", compression="infer")
    test_features.to_pickle("test.pkl.gz", compression="infer")

    print(test_features.head())
    print(test_features.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", nargs="?", default="run", choices=["run", "build", "nn"])
    parser.add_argument("--train", default=False, type=bool)
    parser.add_argument("--method", choices=LEARNING_METHODS)
    parser.add_argument("--pca", type=int)

    args = parser.parse_args()

    if args.action == "run":
        run_main(args)
    elif args.action == "build":
        build_main_dataset(args)
    elif args.action == "nn":
        do_nn(args)
    else:
        raise AssertionError("Unsupported action argument.")
