import numpy as np

class Activation_Softmax:

    def forward(self, inputs):
        exp_value = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_value / np.sum(exp_value, axis=1, keepdims=True)
        self.output = probs
