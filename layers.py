import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

n_inputs = 2
n_neurons = 4

w = 0.01 * np.random.randn(n_inputs, n_neurons)
b = np.zeros((1, n_neurons))

print(f"W:\n{w}\nB:\n{b}")
