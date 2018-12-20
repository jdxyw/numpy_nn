from __future__ import print_function
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(dz):
    return sigmoid(1.0 - sigmoid(dz))


def tanh(z):
    return np.tanh(z)


def tanh_grad(dz):
    return 1 - np.square(np.tanh(dz))
