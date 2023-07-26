import numpy as np


class NeuralNetwork:
    """
    A class used to represent a single neural network

    Attributes:
        weights (list): A list containing a list of list weights of individual neurons
        biases (list): A list of lists containing biases of individual neurons
    """

    class __Layer:
        """
        A class used to represent a single layer

        Attributes:
            weights (list): A list of lists containing weights of individual neurons
            biases (list): A list containing biases of individual neurons
        """

        class __Neuron:
            """
            A class used to represent a single neuron

            Attributes:
                weights (list): (class attribute) A list containing weights of neuron
                bias (float): (class attribute) Neuron's bias
            """

            __slots__ = ['weights', 'bias']

            def __init__(self, no_inputs: int) -> None:
                """
                Creates a single neuron

                Args:
                    no_inputs (int) - number of neuron's inputs
                """

                self.weights = 1 * np.random.randn(no_inputs)
                self.bias = 0.0

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

        def __init__(self, no_inputs: int, no_neurons: int) -> None:
            """
            Creates a layer of neurons

            Args:
                no_inputs (list): A list of numbers of neurons inputs
                no_neurons (list): A list of numbers of neurons in layer
            """

            self.__neurons = []
            for neuron in range(no_neurons):
                self.__neurons.append(self.__Neuron(no_inputs))

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

    def __init__(self, layer_no_inputs: list, layer_no_neurons: list) -> None:
        """
        Creates neural network's layers

        Args:
            no_inputs (list): A list containing numbers of neuron's inputs
            no_neurons (list): A list containing numbers of layer's neurons
        """

        self.__layers = []
        for no_inputs, no_neurons in zip(layer_no_inputs, layer_no_neurons):
            self.__layers.append(self.__Layer(no_inputs, no_neurons))

    def forward(self, inputs: list) -> list:
        """
        Computes a list of layer's outputs

        Args:
            inputs (list): A batch of layer's inputs

        Returns:
            list: A batch of layer's outputs
        """
        outputs_list = []
        for layer_inputs in inputs:
            output = []
            for layer in self.__layers:
                output = layer.forward(layer_inputs)
                layer_inputs = output
            outputs_list.append(output)

        return outputs_list
