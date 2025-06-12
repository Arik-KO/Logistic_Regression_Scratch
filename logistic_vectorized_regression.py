import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gradient_descent import gradient_descent
from compute_cost import compute_cost
from compute_gradient import compute_gradient
from predict_fucntion import predict_function

data = pd.read_csv('ex2data1.txt')
x1 = data.x1.values
x2 = data.x2.values
x_train = np.column_stack((x1, x2))
y_train = data.y.values
print('type of x element is:', type(x_train))
print('first 5 elements of x_train: \n', x_train[:5])
print('type of y element is:', type(y_train))
print('first 5 elements of y_train: \n', y_train[:5])

w, b, j, _ = gradient_descent(x_train, y_train, np.zeros((x_train.shape[1])), 0, compute_cost, compute_gradient, 0.001,
                              1000)
plt.plot(np.arange(1000), j, c='b')
np.random.seed(1)

tmp_p = predict_function(x_train,w,b)
print('the predicted output are: \n',tmp_p[:10])
print('The output given in data: \n',y_train[:10])
plt.title('Iteration vs Cost Function')
plt.xlabel('Number of Iteration')
plt.ylabel('Cost Function Value')
plt.show()
