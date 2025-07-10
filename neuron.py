import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5], 
         [2.0, 5.0, -1.0, 2.0], 
         [-1.5, 2.7, 3.3, -0.8]]

w      = [[0.2, 0.8, -0.5, 1], 
         [0.5, -0.91, 0.26, -0.5], 
         [-0.26, -0.27, 0.17, 0.87]]

b      = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(w).T) + b
o1 = np.dot(inputs, np.array(w).T) # without bias being added
print(f"o1: \n{o1} \n output: \n{outputs}")

