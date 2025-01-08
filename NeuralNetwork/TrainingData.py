import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()


# Dense layer (or fully connected) class
###################

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialze weights and biases
        # Either random params, or load a pre-trained model's finished parameters
        # random weights, centered around 0. [-1, +1]. We multiply with 0.01 to make them a few magintudes smaller to avoid elongated training times. The idea is to have non-zero numbers but small enough that they do not affect training
        # dimension order defined as input, neuron to avoid having to transpose
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # zero bias, most common but not necessary

        self.output = None

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create dense layer with 2 input features (x and y coordinates) and 3 output values (neurons)
dense1 = Layer_Dense(2, 3)

# Perform a foward pass of training data through this layer
dense1.forward(X)

# Output of first 5 samples
print(dense1.output[:5])

