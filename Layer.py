class Layer:
    """
    A class used to represent a single layer

    Attributes:
        neurons (list): (class attribute) A list of layer's neuron objects
        weights (list): A list of lists containing weights of individual neurons
        biases (list): A list containing biases of individual neurons
    """

    __slots__ = ['neurons']

    def __init__(self, weights: list, biases: list) -> None:
        """
        Creates a layer of neurons

        Args:
            weights (list): A list of lists containing weights of individual neurons
            biases (list): A list containing biases of individual neurons
        """

        self.neurons = []
        for neuron_weights, neuron_bias in zip(weights, biases):
            self.neurons.append(self.Neuron(neuron_weights, neuron_bias))

    def forward(self, inputs: list) -> list:
        """
        Computes a list of layer's outputs

        Args:
            inputs (list): A list of layer's inputs

        Returns:
            list: A list of layer's outputs
        """

        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs

    class Neuron:
        """
        A class used to represent a single neuron

        Attributes:
            weights (list): (class attribute) A list containing weights of neuron
            biases (float): (class attribute) Neuron's bias
        """

        __slots__ = ['weights', 'bias']

        def __init__(self, weights: list, bias: float) -> None:
            """
            Creates a single neuron

            Args:
                weights (list): A list containing weights of neuron
                biases (float): Neuron's bias
            """

            self.weights = weights
            self.bias = bias

        def forward(self, inputs: list) -> float:
            """
            Computes a neuron's output

            Args:
                inputs (list): A list of neuron's inputs

            Returns:
                float: Calculated neuron's output
            """

            output = 0.0
            for neuron_weight, neuron_input in zip(self.weights, inputs):
                output += neuron_weight * neuron_input
            output += self.bias
            return output
