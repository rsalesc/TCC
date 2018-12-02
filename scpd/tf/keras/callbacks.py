from keras.callbacks import Callback
from keras.utils import Sequence

import keras

import tensorflow as tf

import numpy as np
import warnings


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
                 best_metric=None,
                 **kwargs):
        self._on_batch = _build_dict(on_batch)
        self._on_epoch = _build_dict(on_epoch)
        self._validation_data = validation_data
        self._best_metric = best_metric
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
                if names[i] == self._best_metric:
                    logs["best_metric"] = res[i]
                logs[names[i]] = res[i]


class SaverCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False,
                 mode='auto', period=1):
        super(SaverCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def save(self, path):
        saver = tf.train.Saver()
        sess = keras.backend.get_session()
        saver.save(sess, "{}.tf".format(path))
        self.model.save(path, overwrite=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.save(filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                self.save(filepath)
