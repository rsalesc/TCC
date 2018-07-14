from keras.callbacks import Callback
from keras.utils import Sequence


def _build_dict(sequence):
    if isinstance(sequence, dict):
        return sequence
    res = {}
    for item in sequence:
        res[item.name()] = item
    return res


class OfflineMetrics(Callback):
    def __init__(self,
                 *args,
                 on_batch=[],
                 on_epoch=[],
                 validation_data=None,
                 **kwargs):
        self._on_batch = _build_dict(on_batch)
        self._on_epoch = _build_dict(on_epoch)
        self._validation_data = validation_data
        super().__init__(*args, **kwargs)

    def set_params(self, params):
        for batch in self._on_batch.keys():
            params["metrics"].extend(batch)
        for epoch in self._on_epoch.keys():
            params["metrics"].extend(epoch)
        super().set_params(params)

    def _iterate_over_validation(self):
        if isinstance(self._validation_data, Sequence):
            for i in range(len(self._validation_data)):
                yield self._validation_data[i]
            return
        # generator
        return self._validation_data()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for metric in self._on_epoch.values():
            metric.reset()

        for x, y in self._iterate_over_validation():
            pred = self.model.predict_on_batch(x)
            for metric in self._on_epoch.values():
                metric.handle_batch(self.model, x, y, pred)

        for names, metric in self._on_epoch.items():
            res = metric.result()
            assert len(res) == len(names)
            for i in range(len(res)):
                logs[names[i]] = res[i]
