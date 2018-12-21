import numpy as np
import matplotlib.pyplot as plt


class Model():
    def __init__(self, hidden_unit=3, input_dim=2, output_dim=2):
        self.params = {}
        self.reg_lambda = 0.01
        self.lr = 0.01
        self.params["w1"] = np.random.randn(input_dim,
                                            hidden_unit) / np.sqrt(input_dim)
        self.params["b1"] = np.random.randn(1, hidden_unit)
        self.params["w2"] = np.random.randn(hidden_unit,
                                            output_dim) / np.sqrt(hidden_unit)
        self.params["b2"] = np.zeros((1, output_dim))

    def forward(self, X, y):
        num_instance = np.shape(X)[0]
        z1 = np.dot(X, self.params["w1"]) + self.params["b1"]
        self.params["z1"] = z1
        a1 = np.tanh(z1)
        self.params["a1"] = a1
        z2 = np.dot(a1, self.params["w2"]) + self.params["b2"]
        self.params["z2"] = z2
        exp_z2 = np.exp(z2)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        self.params["probs"] = probs
        logprobs = -np.log(probs[range(num_instance), y])
        loss = np.sum(logprobs)
        loss += self.reg_lambda / 2 * (
            np.sum(self.params["w1"]) + np.sum(self.params["w2"]))
        return loss / num_instance

    def predict(self, x):
        z1 = np.dot(x, self.params["w1"]) + self.params["b1"]
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.params["w2"]) + self.params["b2"]
        exp_z2 = np.exp(z2)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def backpropagation(self, X, y):
        pred = self.params["probs"]
        num_instance = np.shape(pred)[0]
        pred[range(num_instance), y] -= 1
        grad_out = pred
        grad2 = grad_out.dot(
            self.params["w2"].T) * (1 - np.power(self.params["a1"], 2))
        dw2 = np.dot(self.params["a1"].T, grad_out)
        db2 = np.sum(grad_out, axis=0, keepdims=True)
        dw1 = np.dot(X.T, grad2)
        db1 = np.sum(grad2, axis=0)

        dw2 += self.reg_lambda * self.params["w2"]
        dw1 += self.reg_lambda * self.params["w1"]

        self.params["w1"] += -self.lr * dw1
        self.params["w2"] += -self.lr * dw2
        self.params["b1"] += -self.lr * db1
        self.params["b2"] += -self.lr * db2


def plot_decision_boundary(X, y, pred_func):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)