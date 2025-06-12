import math


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m, n = x.shape
    j_history = []
    w_history = []

    for i in range(num_iters):
        db, dw = gradient_function(x, y, w_in, b_in)

        w_in = w_in - dw * alpha
        b_in = b_in - db * alpha

        if i < 100000:
            cost = cost_function(x, y, w_in, b_in)
            j_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return w_in, b_in, j_history, w_history
