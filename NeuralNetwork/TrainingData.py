import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()


# Dense layer (or fully connected, every neuron in layer n to every neuron in layer n+1) class
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


class Activation_ReLU:

    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # Get normalized probabilities
        # We subtract the largest value to combat exploding values. This makes the range from some negative value up to 0. Normalization allows this
        # The exponential nature of this activation function is why scaling inputs is very important
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Temporary loss class for cross-entropy
# Inherits Loss class
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip both sides to prevent division by 0 and not to drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - for hot-encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create dense layer with 2 input features (x and y coordinates) and 3 output values (neurons)
dense1 = Layer_Dense(2, 3)

# Create ReLU activation
activation1 = Activation_ReLU()

# Second layer with 3 input features and 3 output features
dense2 = Layer_Dense(3, 3)

# Softmax activation
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Perform a foward pass of training data through this layer
dense1.forward(X)

# Forward pass through activation function, takes output from previous layer
activation1.forward(dense1.output)

# Forward pass through the second layer
dense2.forward(activation1.output)

# Forward pass through activation function to get the output of layer 2
activation2.forward(dense2.output)

# Output of first 5 samples
print(activation2.output[:5])

# At this point, the model is random. The classes are predicted with equal probability due to the random normally distributed weights and zeroed biases.
# Need the loss function to determine how wrong the neural network is, and begin adjusting the weights and biases

# Perform forward pass through loss function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)

# Print loss values
print('Loss: ', loss)



