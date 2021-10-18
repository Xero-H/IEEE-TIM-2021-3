import numpy as np
from collections import Counter
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
