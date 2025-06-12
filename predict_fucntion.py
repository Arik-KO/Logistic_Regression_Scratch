import numpy as np
from sigmoid_function import sigmoid


def predict_function(x, w, b):
    z = np.dot(x, w.T) + b
    A = sigmoid(z)
    m, n = x.shape
    p = np.zeros(m)
    for i in range(m):
        p[i] = A[i] >= 0.6
    return p
