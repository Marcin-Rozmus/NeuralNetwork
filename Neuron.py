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

