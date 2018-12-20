import numpy as np


def fc(z, W, b):
    return np.dot(z, W) + b


def fc_bp(ndz, W, z):
    n = z.shape[0]
