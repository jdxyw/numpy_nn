import numpy as np


class Model():
    def __init__(self):
        self.params = {}
        self.reg_lambda = 0

    def forward(self, X, y):
        num_instance = np.shape(X)[0]
        z1 = np.dot(X, self.params["w1"]) + self.params["b1"]
        a1 = np.tanh(z1)
        z2 = np.dot(z1, self.params["w2"]) + self.params["b2"]
        exp_z2 = np.exp(z2)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        logprobs = -np.log(probs[range(num_instance), y])
        loss = np.sum(logprobs)
        loss += reg_lambda / 2 * (
            np.sum(self.params["w1"]) + np.sum(self.params["w2"]))
        return loss / num_instance

    def predict(self, x):
        z1 = np.dot(X, self.params["w1"]) + self.params["b1"]
        a1 = np.tanh(z1)
        z2 = np.dot(z1, self.params["w2"]) + self.params["b2"]
        exp_z2 = np.exp(z2)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
