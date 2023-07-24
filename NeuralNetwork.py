class NeuralNetwork:
    """
    A class used to represent a single neural network

    Attributes:
        weights (list): A list containing a list of list weights of individual neurons
        biases (list): A list of lists containing biases of individual neurons
    """

    class Layer:
        """
        A class used to represent a single layer

        Attributes:
            weights (list): A list of lists containing weights of individual neurons
            biases (list): A list containing biases of individual neurons
        """

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

        __slots__ = ['__neurons']

        def __init__(self, weights: list, biases: list) -> None:
            """
            Creates a layer of neurons

            Args:
                weights (list): A list of lists containing weights of individual neurons
                biases (list): A list containing biases of individual neurons
            """

            self.__neurons = []
            for neuron_weights, neuron_bias in zip(weights, biases):
                self.__neurons.append(self.Neuron(neuron_weights, neuron_bias))

        def forward(self, inputs: list) -> list:
            """
            Computes a list of layer's outputs

            Args:
                inputs (list): A batch of layer's inputs

            Returns:
                list: A batch of layer's outputs
            """
            outputs = []
            for neuron in self.__neurons:
                outputs.append(neuron.forward(inputs))

            return outputs

    __slots__ = ['__layers']

    def __init__(self, weights: list, biases: list) -> None:
        """
        Creates neural network's layers

        Args:
            weights (list): A list of lists containing weights of individual neurons
            biases (list): A list containing biases of individual neurons
        """

        self.__layers = []
        for layer_neurons_weights, layer_neurons_biases in zip(weights, biases):
            self.__layers.append(self.Layer(layer_neurons_weights, layer_neurons_biases))

    def forward(self, inputs: list) -> list:
        """
        Computes a list of layer's outputs

        Args:
            inputs (list): A batch of layer's inputs

        Returns:
            list: A batch of layer's outputs
        """
        outputs_list = []
        for input in inputs:
            output = []
            layer_inputs = input
            for layer in self.__layers:
                output = layer.forward(layer_inputs)
                layer_inputs = output
            outputs_list.append(output)

        return outputs_list
