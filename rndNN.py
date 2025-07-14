import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:

    def forward(self, inputs):

        self.output = np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probs

class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

class CrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,
            axis=1)


        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods


softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
# print(softmax.output)

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
act1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
act2 = Activation_Softmax()
loss_func = CrossEntropy()

# input into first hidden
dense1.forward(X)
act1.forward(dense1.output)

# input into output layer
dense2.forward(act1.output)
act2.forward(dense2.output)

# output

print(act2.output[:5])

loss = loss_func.calculate(act2.output, y)

print(f"Loss: {loss}")

predictions = np.argmax(act2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print("Acc:", accuracy)
