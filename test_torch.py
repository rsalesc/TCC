import torch
import numpy as np

import scpd.torch.lstm as lstm


def optimizer_fn(p):
    return torch.optim.SGD(p, lr=1e-3)


if __name__ == "__main__":
    model = lstm.TripletLSTM(alphabet=4, optimizer_fn=optimizer_fn)

    X = np.array([
        [[1, 3], [1, 2], [0, 1]],
        [[1, 1]],
        [[1, 2]]
    ])
    y = np.array([0, 1, 0])

    for i in range(10000):
        print(model.train_on_batch(X, y)[1])
