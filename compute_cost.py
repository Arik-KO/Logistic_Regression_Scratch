import numpy as np
from sigmoid_function import sigmoid


def compute_cost(x, y, w, b):
    m, _ = x.shape
    z = np.dot(x, w.T) + b
    A = sigmoid(z)
    loss = (-y * np.log(A) - (1 - y) * np.log(1 - A))
    total_cost = np.sum(loss) / m
    return total_cost
