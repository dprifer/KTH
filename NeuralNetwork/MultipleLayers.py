import numpy as np

inputs = [[1, 2, 3, 2.5],           # sample 1
          [2, 5, -1, 2],            # sample 2
          [-1.5, 2.7, 3.3, -0.8]]   # sample 3

weights = [[0.2, 0.8, -0.5, 1],         # Neuron 1
           [0.5, -0.91, 0.26, -0.5],    # Neuron 2
           [-0.26, -0.27, 0.17, 0.87]]  # Neuron 3
biases = [2, 3, 0.5]  # Layer 1

weights2 = [[0.1, -0.14, 0.5],        # Neuron 1
           [-0.5, 0.12, -0.33],       # Neuron 2
           [-0.44, 0.73, 0.13]]       # Neuron 3
biases2 = [-1, 2, -0.5]  # Layer 2


layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
