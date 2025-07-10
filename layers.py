import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()

"""
i = [[1.0, 2.0, 3.0, 2.5], 
     [2.0, 5.0, -1.0, 2.0], 
     [-1.5, 2.7, 3.3, -0.8]]

w1 = [[0.2, 0.8, -0.5, 1], 
     [0.5, -0.91, 0.26, -0.5], 
     [-0.26, -0.27, 0.17, 0.87]]

w2 = [[0.1, -0.14, 0.5],
      [-0.5, 0.12, -0.33],
      [-0.44, 0.73, -0.13]]

b1  = [2.0, 3.0, 0.5]
b2 = [-1.0, 2.0, -0.5]

l1_out = np.dot(i, np.array(w1).T) + b1
l2_out = np.dot(l1_out, np.array(w2).T) + b2

print(l2_out)
"""
