import math

import numpy as np


inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]


# ReLU
# For nonlinear relationships, e.g. y = sin(x)
##############################################

output = []
for i in inputs:
    output.append(max(0,i))

print(output)

# With NumPy
output = np.maximum(0, inputs)
print(output)


# Softmax
# For classification
# takes in non-normalized, or uncalibrated, inputs and produce a normalized distribution of probabilities for our classes
# represents confidence scores for each class that adds up to 1. Prediction is the neuron with the highest confidence score
########################################################################################################################

layer_outputs = [4.8, 1.21, 2.385]

# Exponentiate outputs, i.e. y = e^x: values become non-negative
E = math.e
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print('Exponential values: ', exp_values)

# Normalize values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized values: ', norm_values)
print('Sum of nomalized values: ', sum(norm_values))

# With NumPy
exp_values = np.exp(layer_outputs)
# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # for batches
norm_values = exp_values / sum(exp_values)  # for single input
print('Normalized values: ', norm_values)
print('Sum of exp values: ', sum(norm_values))

inputs = [[-2, -1, 0]]
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
# Normalize them for each sample
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print('Probabilities: ', probabilities)