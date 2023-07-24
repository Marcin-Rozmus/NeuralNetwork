from pprint import pprint

import NeuralNetwork

# batch of inputs
nn_inputs = [[1, 2, 3],
             [2, 3, 4],
             [-1, -1, -1.3],
             [-2.9, -3.2, -4]]

# 3 weights for 2 neurons
neurons_weights = [[[1.0, 0.5, -0.2], [1.3, 0.8, 0.5]], ]
# biases for 2 neurons
neuron_biases = [[1.5, 2], ]

# neural network creation
nn = NeuralNetwork.NeuralNetwork(neurons_weights, neuron_biases)
# compute neural network output
nn_output = nn.forward(nn_inputs)

# print results
print(nn_output)
