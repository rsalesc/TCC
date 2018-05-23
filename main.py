import random
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scpd import utils
from scpd.cf import (HTTP_POOL, DiskCodeExtractor, ParticipantExtractor,
                     BatchSubmissionExtractor, CODEFORCES_POOL,
                     get_cached_rated_list, FilterProvider)
from scpd.metrics import BinaryClassificationMetrics
from scpd.source import SourceCodePairing
from scpd.features.base import (BaseFeatureExtractor,
                                LabelerPairFeatureExtractor)

MAGICAL_SEED = 42
SUBMISSION_API_COUNT = 10000


class DatasetBuilder():
    def __init__(self, training_size, test_size, training_file, test_file,
                 submissions_per_user):
        self._participant_extractor = ParticipantExtractor(
            get_cached_rated_list())
        self._training_size = training_size
        self._test_size = test_size
        self._training_file = training_file
        self._test_file = test_file
        self._submissions_per_user = submissions_per_user

    def fetch_dataset(self, K):
        participants = self._participant_extractor.extract(K)
        submission_extractor = BatchSubmissionExtractor(
            CODEFORCES_POOL, participants, count=SUBMISSION_API_COUNT)
        return submission_extractor.extract(
            FilterProvider().filter(), limit=self._submissions_per_user)

    def extract(self, force=False):
        should_fetch = (force or not os.path.isfile(self._training_file)
                        or not os.path.isfile(self._test_file))
        training_submissions = None
        test_submissions = None

        if should_fetch:
            training_submissions = self.fetch_dataset(self._training_size)
            test_submissions = self.fetch_dataset(self._test_size)

            with utils.opens(self._training_file, "wb") as f:
                pickle.dump(training_submissions, f)
            with utils.opens(self._test_file, "wb") as f:
                pickle.dump(test_submissions, f)

        if training_submissions is None:
            training_submissions = pickle.load(
                utils.opens(self._training_file, "rb"))
        if test_submissions is None:
            test_submissions = pickle.load(utils.opens(self._test_file, "rb"))

        return training_submissions, test_submissions


def extract_codes(submissions):
    code_extractor = DiskCodeExtractor(HTTP_POOL, submissions)
    return code_extractor.extract()


def make_pairs(sources, k1, k2):
    pair_maker = SourceCodePairing()
    return pair_maker.make_pairs(sources, k1=k1, k2=k2)


def extract_features(pairs):
    extractor = LabelerPairFeatureExtractor(BaseFeatureExtractor())
    return extractor.extract(pairs)


def split_label(df):
    if "label" not in df:
        return df, None
    feature_matrix = df.drop(columns=["label"]).values
    label_array = df["label"].values
    return feature_matrix, label_array


class RandomForestMethod():
    def __init__(self):
        self._classifier = RandomForestClassifier(
            n_estimators=100, random_state=MAGICAL_SEED)

    def fit(self, df):
        feature_matrix, label_array = split_label(df)
        self._classifier.fit(feature_matrix, label_array)

    def predict(self, df):
        feature_matrix, label_array = split_label(df)
        label_pred = self._classifier.predict(feature_matrix)
        return label_pred, label_array


if __name__ == "__main__":
    random.seed(MAGICAL_SEED)

    builder = DatasetBuilder(
        training_size=500,
        test_size=75,
        training_file="submissions_training.dat",
        test_file="submissions_test.dat",
        submissions_per_user=10)
    training_submissions, test_submissions = builder.extract()
    training_sources, test_sources = extract_codes(
        training_submissions), extract_codes(test_submissions)

    random.seed(MAGICAL_SEED * 2)
    training_pairs = make_pairs(training_sources, k1=10000, k2=10000)
    test_pairs = make_pairs(test_sources, k1=1000, k2=1000)

    training_features, test_features = extract_features(
        training_pairs), extract_features(test_pairs)

    # Random Forest
    random_forest = RandomForestMethod()
    random_forest.fit(training_features)
    test_pred, test_label = random_forest.predict(test_features)

    metrics = BinaryClassificationMetrics(test_pred, test_label)
    print(metrics)
