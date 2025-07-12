import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
[0, 1, 0],
[0, 1, 0]])

# for targ_idx, distrb in zip(target_output, softmax_outputs):
    # print(distrb[targ_idx])

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]

elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs * class_targets,
                                 axis=1)

n_log = -np.log(correct_confidences)
avg_loss = np.mean(n_log)                                
print(avg_loss)

print(np.e**(-np.inf))
