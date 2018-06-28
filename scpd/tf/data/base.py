class BaseBatchProvider:
    def __init__(self, *args, **kwargs):
        pass

    def next_batch(self, batch_size):
        raise NotImplementedError()
