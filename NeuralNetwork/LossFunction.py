"""
Different types are:
- mean squared error: used for regression
- categorical crossentropy: used for classification probles, most common with softmax activation function on the output layer
  compares ground truth probability (y as a one-hot vector) with some predicted distribution (y-hat)
"""

import math
import numpy as np

## Categorical crossentropy

# An example output from the output layer of the NN
softmax_output = [0.7, 0.1, 0.2]
# Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)

# The same result is achieved with just
loss = -math.log(softmax_output[0])
print(loss)


# Probabilities for 3 samples (working on batches of inputs)
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]  # dog, cat, cat

# Printing the confidence corresponding to the one-hot index
for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])

# Or with NumPy
# first parameter is the row indices, second is the column index
print(softmax_outputs[[0, 1, 2], class_targets])

# Dynamically defined as the row indices are always [0 n-1] for n rows
print(softmax_outputs[range(len(softmax_output)), class_targets])

# Apply the negative log
print(-np.log(softmax_outputs[range(len(softmax_output)), class_targets]))

# We also want to average our loss per batch to know how our model is doing during training
neg_log = -np.log(softmax_outputs[range(len(softmax_output)), class_targets])
average_loss = np.mean(neg_log)
print(average_loss)


# Class targets can come sparsely ([0, 2, 1] or as one-hot encoded ([[1,0,0], [0,0,1], [0,1,0]]) so we implement a test
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

#Probabilities for target values - only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]

# Mask values - only for one-hot encoded
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs * class_targets, axis=-1)

# Losses
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print(average_loss)

# When the prediction confidence is 0, np.log(0) would fail, so we need a workaround. We cannot just set an exception for
# correct_confidences = 0 --> neg_log = -np.inf as an infinitely small number would mess up the average loss calculation
# adding a small number to the confidences, e.g. 1e-7 would lead to a different problem as the -log of a number >1 will be negative
# The best practice is to clip values from both ends, i.e. the largest confidence should be 1-1e-7 and the smallest confidence 1e-7
softmax_outputs = np.array([[0, 0, 1],
                            [0, 1, 0],
                            [0.02, 0.9, 0.08]])

softmax_outputs_clipped = np.clip(softmax_outputs, a_min=1e-7, a_max=1-1e-7)
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs_clipped[range(len(softmax_outputs_clipped)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs_clipped * class_targets, axis=-1)

neg_log = -np.log(correct_confidences)
print(neg_log)
average_loss = np.mean(neg_log)
print(average_loss)