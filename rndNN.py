import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
from layer_dense import Layer_Dense
from ReLU import Activation_ReLU
from softmax import Activation_Softmax
from crossEntr import Loss, CrossEntropy

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
act1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
act2 = Activation_Softmax()
loss_func = CrossEntropy()

# helper variables
lowest_loss = 9999999
best_dense1_w = dense1.weights.copy()
best_dense1_b = dense1.bias.copy()
best_dense2_w = dense2.weights.copy()
best_dense2_b = dense2.bias.copy()

for iteration in range(10000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.bias += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.bias += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    act1.forward(dense1.output)
    dense2.forward(act1.output)
    act2.forward(dense2.output)

    loss = loss_func.calculate(act2.output, y)

    predictions = np.argmax(act2.output, axis=1)
    accuracy = np.mean(predictions==y)

    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
        'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_bias = dense1.bias.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_bias = dense2.bias.copy()
        lowest_loss = loss

    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.bias = best_dense1_bias.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.bias = best_dense2_bias.copy()
