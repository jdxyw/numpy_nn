import numpy as np


def mse(pred, y):
    loss = np.mean(np.sum(np.square(pred - y), axis=-1))
    grad = pred - y
    return loss, grad
