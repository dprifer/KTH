import numpy as np

inputs = [1, 2, 3, 2.5]

# initialize weights randomly
weights = [0.2, 0.8, -0.5, 1]

# a single neuron only needs one bias value as there is one bias per neuron
bias = 2

### Plain Python

output = (inputs[0]*weights[0] +
          inputs[1]*weights[1] +
          inputs[2]*weights[2] +
          inputs[3]*weights[3] + bias)

print(output)


#########
### NumPy

outputs = np.dot(weights, inputs) + bias

print(output)
