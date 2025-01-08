import numpy as np

inputs = [1, 2, 3, 2.5]  # Feature set for one observation

# initialize weights and biases randomly
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

### Plain Python

outputs = [
    # Neuron 1
          inputs[0]*weights1[0] +
          inputs[1]*weights1[1] +
          inputs[2]*weights1[2] +
          inputs[3]*weights1[3] + bias1,

    # Neuron 2
          inputs[0]*weights2[0] +
          inputs[1]*weights2[1] +
          inputs[2]*weights2[2] +
          inputs[3]*weights2[3] + bias2,
    # Neuron 3
          inputs[0]*weights3[0] +
          inputs[1]*weights3[1] +
          inputs[2]*weights3[2] +
          inputs[3]*weights3[3] + bias3]

print(outputs)


## We can also use a loop to do the same

weights = [[0.2, 0.8, -0.5, 1],         # Neuron 1
           [0.5, -0.91, 0.26, -0.5],    # Neuron 2
           [-0.26, -0.27, 0.17, 0.87]]  # Neuron 3

biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += weight * n_input
    # Add bias
    neuron_output += neuron_bias
    # Put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)

print(layer_outputs)


#########
### NumPy

# Qpplied theories: dot product, vector addition
layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)


# Adding a batch of data
# When inputs is a vector and weights is matrix --> dot product can be used that results in a vector
# When input is also a matrix --> dot product on all vectors from input and weight matrices --> matrix product
# It takes all combination of rows and columns. To make them compatible, weight matrix must be transposed

inputs = [[1, 2, 3, 2.5],           # sample 1
          [2, 5, -1, 2],            # sample 2
          [-1.5, 2.7, 3.3, -0.8]]   # sample 3

# The output matrix consists of all atomic dot products, i.e. the outputs of all neurons after each sample
outputs = np.dot(inputs, np.array(weights).T) + biases
# We want to have a list of layer outputs per each sample than a list of neurons and their sample-wise outputs --> weight is the second term

print(outputs)


