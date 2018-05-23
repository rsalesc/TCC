import numpy as np


class BinaryClassificationMetrics():
    def __init__(self, predicted, expected):
        self._predicted = np.array(predicted)
        self._expected = np.array(expected)

    def instances(self):
        return len(self._predicted)

    def condition_positive(self):
        return np.sum(self._expected)

    def condition_negative(self):
        return np.sum(1 - self._expected)

    def predicted_positive(self):
        return np.sum(self._predicted)

    def predicted_negative(self):
        return np.sum(1 - self._predicted)

    def true_positive(self):
        return np.sum(self._predicted * self._expected)

    def true_negative(self):
        return np.sum((1 - self._predicted) * (1 - self._expected))

    def false_positive(self):
        return np.sum(self._predicted * (1 - self._expected))

    def false_negative(self):
        return np.sum((1 - self._predicted) * self._expected)

    def ok(self):
        return self.true_positive() + self.true_negative()

    def fail(self):
        return self.false_positive + self.false_negative()

    def accuracy(self):
        return self.ok() / self.instances()

    def recall(self):
        return self.true_positive() / self.condition_positive()

    def precision(self):
        return self.true_positive() / self.predicted_positive()

    def fallout(self):
        return self.false_positive() / self.condition_negative()

    def f1(self):
        return 2.0 / ((1.0 / self.recall()) + (1.0 / self.precision()))

    def __str__(self):
        res = []
        res.append("Instances evaluated: {:7}".format(self.instances()))
        res.append("Accuracy: {:7.4f}".format(self.accuracy()))
        res.append("Recall: {:7.4f}".format(self.recall()))
        res.append("Precision: {:7.4f}".format(self.precision()))
        res.append("Fall-out: {:7.4f}".format(self.fallout()))
        res.append("F1 score: {:7.4f}".format(self.f1()))
        return "\n".join(res)
