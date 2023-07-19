class Layer:
    __slots__ = ['neurons']

    def __init__(self, weights: list, biases: list) -> None:
        self.neurons = []
        for neuron_weights, neuron_bias in zip(weights, biases):
            self.neurons.append(self.Neuron(neuron_weights, neuron_bias))

    def forward(self, inputs: list) -> list:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs

    class Neuron:
        __slots__ = ['weights', 'bias']

        def __init__(self, weights: list, bias: float) -> None:
            self.weights = weights
            self.bias = bias

        def forward(self, inputs: list) -> float:
            output = 0.0
            for neuron_weight, neuron_input in zip(self.weights, inputs):
                output += neuron_weight * neuron_input
            output += self.bias
            return output
