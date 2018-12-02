import tensorflow as tf
import numpy as np


def safe_nanmax(x):
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore',
                                   r'All-NaN (slice|axis) encountered')
        return np.nanmax(x)


def safe_nanargmax(x):
    try:
        return np.nanargmax(x)
    except ValueError:
        return np.nan


def upper_triangular_flat(A):
    ones = tf.ones_like(A)
    mask_a = tf.matrix_band_part(ones, 0, -1)
    mask_b = tf.matrix_band_part(ones, 0, 0)
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)

    return tf.boolean_mask(A, mask)


def pairwise_distances(embeddings, embeddings_b=None, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    if embeddings_b is not None:
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings_b))
    else:
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * \
        dot_product + tf.expand_dims(square_norm, 0)

    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def pairwise_nm_distances(A, embeddings_b, scope=None, squared=False):
    """
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      distances,    [m,n] matrix of pairwise distances
    """
    with tf.variable_scope('pairwise_dist'):
        B = embeddings_b

        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        distances = tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0)

        if not squared:
            mask = tf.to_float(tf.equal(distances, 0.0))
            distances = distances + mask * 1e-16
            distances = tf.sqrt(distances)
            distances = distances * (1.0 - mask)

        return distances


def contrastive_score(labels, dist, thresholds, metric="accuracy"):
    d = {}
    if isinstance(metric, list):
        for m in metric:
            d[m] = True
    else:
        d[metric] = True
    res = {}

    if "total" in d:
        res["total"] = tf.size(labels)
    if "f1" in d:
        precision = contrastive_score(
            labels, dist, thresholds, metric="precision")
        recall = contrastive_score(labels, dist, thresholds, metric="recall")
        res["f1"] = 2 * precision * recall / (precision + recall)
    if "bacc" in d:
        specificity = contrastive_score(
            labels, dist, thresholds, metric="specificity")
        recall = contrastive_score(labels, dist, thresholds, metric="recall")
        res["metric"] = (specificity + recall) / 2

    th = tf.reshape(thresholds, [1, -1])
    dist = tf.reshape(dist, [-1, 1])

    labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.int32)
    pred = tf.cast(dist < th, tf.int32)

    tp = pred * labels
    tn = (1 - pred) * (1 - labels)
    corr = tp + tn

    total = tf.cast(tf.size(labels), tf.float32)
    tp = tf.reduce_sum(tf.cast(tp, tf.float32), axis=0)
    tn = tf.reduce_sum(tf.cast(tn, tf.float32), axis=0)
    pred = tf.cast(pred, tf.float32)
    corr = tf.cast(corr, tf.float32)
    labels = tf.cast(labels, tf.float32)

    if "accuracy" in d:
        res["accuracy"] = tf.reduce_mean(corr, axis=0)
    if "precision" in d:
        res["precision"] = tp / tf.reduce_sum(pred, axis=0)
    if "recall" in d:
        res["recall"] = tp / tf.reduce_sum(labels)
    if "specificity" in d:
        res["specificity"] = tn / tf.reduce_sum(1 - labels)
    if "tp" in d:
        res["tp"] = tp
    if "tn" in d:
        res["tn"] = tn
    if "pcp" in d:
        res["pcp"] = tf.reduce_sum(pred, axis=0)
    if "pcn" in d:
        res["pcn"] = tf.reduce_sum(1 - pred, axis=0)
    if "cp" in d:
        res["cp"] = tf.reduce_sum(labels)
    if "cn" in d:
        res["cn"] = tf.reduce_sum(1 - labels)
    if "eer" in d:
        far = (tf.reduce_sum(pred, axis=0) - tp) / total
        frr = (tf.reduce_sum(1 - pred, axis=0) - tn) / total
        argmin = tf.argmin(tf.abs(far - frr), axis=-1)
        res["eer"] = tf.gather((far + frr) / 2, argmin, axis=-1)

    if len(d) != len(res):
        raise NotImplementedError("some metrics were not implemented")
    if not isinstance(metric, list):
        return next(iter(res.values()))
    return res


def cross_score(labels_a, embeddings_a, labels_b, embeddings_b,
                thresholds, metric="accuracy", dist_fn=pairwise_distances):
    dist = dist_fn(embeddings_a, embeddings_b=embeddings_b)
    labels_a = tf.reshape(labels_a, [-1, 1])
    labels_b = tf.reshape(labels_b, [-1, 1])
    pair_labels = tf.cast(tf.equal(labels_a, tf.transpose(labels_b)), tf.int32)
    flat_labels = tf.reshape(pair_labels, [-1, 1])
    flat_dist = tf.reshape(dist, [-1, 1])

    return contrastive_score(flat_labels, flat_dist, thresholds, metric=metric)


def triplet_score(labels, embeddings, thresholds, metric="accuracy"):
    dist = pairwise_distances(embeddings)
    labels = tf.reshape(labels, [-1, 1])
    pair_labels = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.int32)
    flat_labels = upper_triangular_flat(pair_labels)
    flat_dist = upper_triangular_flat(dist)

    return contrastive_score(flat_labels, flat_dist, thresholds, metric=metric)


def categorical_score(y_true, y_pred, metric="accuracy"):
    d = {}
    if isinstance(metric, list):
        for m in metric:
            d[m] = True
    else:
        d[metric] = True
    res = {}

    labels = tf.argmax(y_true, axis=-1)
    labeled = tf.argmax(y_pred, axis=-1)

    corr = tf.equal(labels, labeled)
    corr = tf.cast(corr, tf.float32)
    total = tf.shape(labels)[0]

    if "total" in d:
        res["total"] = total
    if "correct" in d:
        res["correct"] = tf.reduce_sum(corr, axis=0)

    if "accuracy" in d:
        res["accuracy"] = tf.reduce_mean(corr, axis=0)

    if len(d) != len(res):
        raise NotImplementedError("some metrics were not implemented")
    if not isinstance(metric, list):
        return next(iter(res.values()))
    return res


class BatchScorer:
    def __init__(self):
        self._tp = 0
        self._tn = 0
        self._pcp = 0
        self._pcn = 0
        self._cp = 0
        self._cn = 0
        self._total = 0

    def score(self, y_true, y_pred, metric):
        raise NotImplementedError()

    def handle(self, y_true, y_pred):
        d = self.score(y_true, y_pred,
                       ["tp", "tn", "pcp", "pcn", "cp", "cn", "total"])
        self._tp += d["tp"]
        self._tn += d["tn"]
        self._pcp += d["pcp"]
        self._pcn += d["pcn"]
        self._cp += d["cp"]
        self._cn += d["cn"]
        self._total += d["total"]

    def result(self, metric):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore")

            if metric == "accuracy":
                return (self._tp + self._tn) / self._total
            if metric == "precision":
                return self._tp / self._pcp
            if metric == "recall":
                return self._tp / self._cp
            if metric == "specificity":
                return self._tn / self._cn
            if metric == "far":
                return (self._pcp - self._tp) / self._total
            if metric == "frr":
                return (self._pcn - self._tn) / self._total
            if metric == "f1":
                precision = self.result("precision")
                recall = self.result("recall")
                return 2 * precision * recall / (precision + recall)
            if metric == "bacc":
                recall = self.result("recall")
                specificity = self.result("specificity")
                return (recall + specificity) / 2
            if metric == "eer":
                far = self.result("far")
                frr = self.result("frr")
                argmin = np.argmin(np.abs(far - frr), axis=-1)
                print(argmin)
                print(far[..., argmin])
                print(frr[..., argmin])
                return np.array((far + frr) / 2)[..., argmin]

        raise NotImplementedError()


class ContrastiveBatchScorer(BatchScorer):
    def __init__(self, margin, *args, **kwargs):
        self._margin = margin
        self._sess = tf.Session()
        super().__init__(*args, **kwargs)

    def score(self, y_true, y_pred, metric):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with graph.as_default():
                return sess.run(
                    contrastive_score(
                        tf.convert_to_tensor(y_true, tf.float32),
                        tf.convert_to_tensor(y_pred, tf.float32),
                        tf.convert_to_tensor(self._margin, tf.float32),
                        metric=metric))


class TripletBatchScorer(ContrastiveBatchScorer):
    def score(self, y_true, y_pred, metric):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with graph.as_default():
                return sess.run(
                    triplet_score(
                        tf.convert_to_tensor(y_true, tf.float32),
                        tf.convert_to_tensor(y_pred, tf.float32),
                        tf.convert_to_tensor(self._margin, tf.float32),
                        metric=metric))


class FlatPairBatchScorer(ContrastiveBatchScorer):
    def score(self, y_true, y_pred, metric):
        assert y_pred.shape[0] == y_true.shape[0] * 2
        a, b = np.split(y_pred, 2)
        dist = np.linalg.norm(a - b, axis=1)
        return super().score(y_true, dist, metric)


class CategoricalBatchScorer(BatchScorer):
    def __init__(self):
        self._correct = 0
        self._total = 0

    def handle(self, y_true, y_pred):
        d = self.score(y_true, y_pred, ["correct", "total"])
        self._correct += d["correct"]
        self._total += d["total"]

    def score(self, y_true, y_pred, metric):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with graph.as_default():
                return sess.run(
                    categorical_score(
                        tf.convert_to_tensor(y_true, tf.float32),
                        tf.convert_to_tensor(y_pred, tf.float32),
                        metric=metric))

    def result(self, metric):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore")

            if metric == "accuracy":
                return self._correct / self._total

        raise NotImplementedError()


class CompletePairContrastiveScorer(TripletBatchScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels = []
        self._embeddings = []

    def handle_finish(self, d):
        self._tp += d["tp"]
        self._tn += d["tn"]
        self._pcp += d["pcp"]
        self._pcn += d["pcn"]
        self._cp += d["cp"]
        self._cn += d["cn"]
        self._total += d["total"]

    def score_cross(self, labels, embeddings, i, metric):
        dist_fn = pairwise_nm_distances
        if labels.shape[0] != self._labels[i].shape[0]:
            dist_fn = pairwise_nm_distances
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with graph.as_default():
                return sess.run(
                    cross_score(
                        tf.convert_to_tensor(labels, tf.float32),
                        tf.convert_to_tensor(embeddings, tf.float32),
                        tf.convert_to_tensor(self._labels[i], tf.float32),
                        tf.convert_to_tensor(self._embeddings[i], tf.float32),
                        tf.convert_to_tensor(self._margin, tf.float32),
                        metric=metric,
                        dist_fn=dist_fn))

    def handle(self, labels, embeddings):
        labels = np.array(labels)
        embeddings = np.array(embeddings)

        METRICS = ["tp", "tn", "pcp", "pcn", "cp", "cn", "total"]

        for i in range(len(self._labels)):
            self.handle_finish(
                self.score_cross(labels, embeddings, i, METRICS))

        self.score(labels, embeddings, METRICS)

        self._labels.append(labels)
        self._embeddings.append(embeddings)
        assert len(self._labels) == len(self._embeddings)

    def result(self, metric):
        return super().result(metric)


class ContrastiveOnKerasMetric:
    def __init__(self, margin, metric="accuracy"):
        self.__name__ = "contrastive_{}".format(metric)
        self._margin = margin
        self._metric = metric

    def __call__(self, labels, embeddings):
        return contrastive_score(
            labels,
            embeddings,
            tf.convert_to_tensor(self._margin),
            metric=self._metric)


class TripletOnKerasMetric:
    def __init__(self, margin, metric="accuracy"):
        self.__name__ = "triplet_{}".format(metric)
        self._margin = margin
        self._metric = metric

    def __call__(self, labels, embeddings):
        return triplet_score(
            labels,
            embeddings,
            tf.convert_to_tensor(self._margin),
            metric=self._metric)


class CategoricalOnKerasMetric:
    def __init__(self, metric="accuracy"):
        self.__name__ = "categorical_{}".format(metric)
        self._metric = metric

    def __call__(self, y_true, y_pred):
        return categorical_score(
            tf.convert_to_tensor(y_true, tf.float32),
            tf.convert_to_tensor(y_pred, tf.float32),
            metric=self._metric)


class OfflineMetric:
    def __init__(self, *args, **kwargs):
        self.__name__ = self.name()

    def name(self):
        raise NotImplementedError()

    def handle_batch(self, model, x, labels, pred):
        raise NotImplementedError()

    def result(self):
        raise NotImplementedError()

    def reset(self):
        pass


class SimilarityValidationMetric(OfflineMetric):
    def __init__(self,
                 margin,
                 *args,
                 id="sim",
                 metric=["accuracy"],
                 argmax=None,
                 **kwargs):
        self._margin = np.array(margin)
        assert argmax is None or (self._margin.ndim == 1 and argmax in metric)
        self._metric = metric if isinstance(metric, list) else [metric]
        self._argmax = argmax
        self._scorer = None
        self._id = id
        super().__init__(self, *args, **kwargs)

    def name(self):
        metrics = list(
            map(lambda x: "val_{}_{}".format(self._id, x), self._metric))
        if self._argmax is not None:
            metrics.append("val_{}_argmax_{}".format(self._id, self._argmax))
        return tuple(metrics)

    def handle_batch(self, model, x, labels, pred):
        self._scorer.handle(labels, pred)

    def result(self):
        if self._argmax is None:
            metrics = map(lambda x: safe_nanmax(self._scorer.result(x)),
                          self._metric)
            return tuple(metrics)
        else:
            argmax = safe_nanargmax(self._scorer.result(self._argmax))
            metrics = map(lambda x: self._scorer.result(x)[argmax],
                          self._metric)
            return tuple(metrics) + (self._margin[argmax], )


class TripletValidationMetric(SimilarityValidationMetric):
    def __init__(self, *args, id="triplet", **kwargs):
        super().__init__(*args, id=id, **kwargs)

    def reset(self):
        self._scorer = TripletBatchScorer(self._margin)


class ContrastiveValidationMetric(SimilarityValidationMetric):
    def __init__(self, *args, id="contrastive", **kwargs):
        super().__init__(*args, id=id, **kwargs)

    def reset(self):
        self._scorer = ContrastiveBatchScorer(self._margin)


class FlatPairValidationMetric(SimilarityValidationMetric):
    def __init__(self, *args, id="fpair", **kwargs):
        super().__init__(*args, id=id, **kwargs)

    def reset(self):
        self._scorer = FlatPairBatchScorer(self._margin)


class CompletePairValidationMetric(SimilarityValidationMetric):
    def __init__(self, *args, id="complete", **kwargs):
        super().__init__(*args, id=id, **kwargs)

    def reset(self):
        self._scorer = CompletePairContrastiveScorer(self._margin)


class CategoricalValidationMetric(OfflineMetric):
    def __init__(self, metric=["accuracy"]):
        self._metric = list(metric)
        self._scorer = None

    def name(self):
        return tuple(["val_{}".format(x) for x in self._metric])

    def handle_batch(self, model, x, labels, pred):
        self._scorer.handle(labels, pred)

    def result(self):
        return tuple([self._scorer.result(x) for x in self._metric])

    def reset(self):
        self._scorer = CategoricalBatchScorer()
