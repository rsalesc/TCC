from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .. import utils


class DataframeFitter():
    def __init__(self, folder=None, random_state=None, *args, **kwargs):
        self._folder = None
        self._random_state = random_state

    def fit_fold(self, x, y):
        raise NotImplementedError()

    def fit(self, df):
        x, y = utils.split_label(df)
        if y is None:
            raise AssertionError("fitting data should contain label field")
        if self._folder is None:
            self.fit_fold(x, y)
        else:
            for train_index, _ in self._folder.split(x, y):
                x_train, y_train = x[train_index], y[train_index]
                self.fit_fold(x_train, y_train)

    def predict(self, df):
        x, y = utils.split_label(df)
        y_pred = self._classifier.predict(x)
        return y_pred, y


class RandomForestFitter(DataframeFitter):
    def __init__(self, folder=None, random_state=None, *args, **kwargs):
        super().__init__(
            self, folder=folder, random_state=random_state, *args, **kwargs)
        self._classifier = RandomForestClassifier(
            random_state=random_state, **kwargs)

    def fit_fold(self, x, y):
        self._classifier.fit(x, y)


class XGBoostFitter(DataframeFitter):
    def __init__(self, folder=None, random_state=None, *args, **kwargs):
        super().__init__(
            self, folder=folder, random_state=random_state, *args, **kwargs)
        self._classifier = XGBClassifier(random_state=random_state, **kwargs)

    def fit_fold(self, x, y):
        self._classifier.fit(x, y)
