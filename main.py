import random
from sklearn.model_selection import StratifiedKFold

from scpd import source
from scpd.features import (LabelerPairFeatureExtractor, BaseFeatureExtractor)
from scpd.metrics import BinaryClassificationMetrics
from scpd.datasets import CodeforcesDatasetBuilder
from scpd.learning.scikit import RandomForestFitter, XGBoostFitter

MAGICAL_SEED = 42
SUBMISSION_API_COUNT = 10000


def extract_features(pairs):
    extractor = LabelerPairFeatureExtractor(BaseFeatureExtractor())
    return extractor.extract(pairs)


if __name__ == "__main__":
    random.seed(MAGICAL_SEED)

    builder = CodeforcesDatasetBuilder(
        training_size=500,
        test_size=75,
        training_file="submissions_training.dat",
        test_file="submissions_test.dat",
        submissions_per_user=10)
    training_sources, test_sources = builder.extract()

    random.seed(MAGICAL_SEED * 2)
    training_pairs = source.make_pairs(training_sources, k1=10000, k2=10000)
    test_pairs = source.make_pairs(test_sources, k1=1000, k2=1000)

    training_features = extract_features(training_pairs)
    test_features = extract_features(test_pairs)

    folder = StratifiedKFold(
        n_splits=7, random_state=MAGICAL_SEED, shuffle=True)

    # Random Forest
    random_forest = XGBoostFitter(
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=5,
        random_state=MAGICAL_SEED,
        folder=folder)
    random_forest.fit(training_features)
    test_pred, test_label = random_forest.predict(test_features)

    metrics = BinaryClassificationMetrics(test_pred, test_label)
    print(metrics)
