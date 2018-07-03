class BaseModel:
    def build(self):
        raise NotImplementedError()

    def train(self, generator, callbacks=None, *args, **kwargs):
        cb = []
        if callbacks is not None:
            cb.extend(callbacks)

        self.model.fit_generator(generator, callbacks=cb, *args, **kwargs)

    def test(self, x, y, batch_size=32, *args, **kwargs):
        self.model.evaluate(x, y, batch_size, *args, **kwargs)
