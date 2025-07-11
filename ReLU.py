import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layer_dense import Layer_Dense

class Activation_ReLU:

    def forward(self, inputs):

        self.output = np.maximum(0, inputs)

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense1.forward(X)
activation1.forward(dense1.output)

print(activation1.output[:5])



