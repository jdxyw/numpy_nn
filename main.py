from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import nn

X, y = sklearn.datasets.make_moons(200, noise=0.20)

iterations = 20000

model = nn.Model()
for i in range(iterations):
    model.forward(X, y)
    model.backpropagation(X, y)

nn.plot_decision_boundary(X, y, lambda X: model.predict(X))
plt.show()