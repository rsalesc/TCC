import tensorflow as tf


class BaseTrainer:
    def __init__(self,
                 sess,
                 model,
                 provider,
                 n_epochs=1,
                 saver=None,
                 global_step_tensor=None,
                 *args,
                 **kwargs):
        self._config = kwargs
        self._n_epochs = n_epochs
        self._sess = sess
        self._model = model
        self._provider = provider
        self._saver = saver

        self._global_step_tensor = global_step_tensor 
        self._model.set_global_step(self._global_step_tensor)

    def should_save(self, *args, **kwargs):
        if self._saver is not None:
            return self._saver.should_save(*args, **kwargs)
        return False

    def save(self):
        if self._saver is not None:
            self._saver.save(global_step_tensor=self._global_step_tensor)

    def train(self):
        for current_epoch in range(
                self._model.epoch(self._sess), self._n_epochs):
            self.train_epoch()
            self._sess.run(self._model.step_epoch)
            self.validate()
            if (self.should_save(current_epoch)
                    or current_epoch + 1 == self._n_epochs):
                self.save()

    def train_epoch(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()
