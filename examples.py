import Layer

inputs = [1, 2, 3]

neurons_weights = [[1.0, 0.5, -0.2],
                   [1.3, 0.8, 0.5]]
neuron_biases = [1.5,
                 2]

layer = Layer.Layer(neurons_weights, neuron_biases)
layer_output = layer.forward(inputs)

print(layer_output)
