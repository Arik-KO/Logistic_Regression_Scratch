from sigmoid_function import sigmoid
import numpy as np


def compute_gradient(x, y, w, b):
    m, n = x.shape
    z = np.dot(x, w.T) + b
    A = sigmoid(z)
    dz = A - y
    db = np.sum(dz)
    dw = np.dot(dz, x)
    db /= m
    dw /= m

    return db, dw
