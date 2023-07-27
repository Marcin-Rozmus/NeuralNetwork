import numpy as np


class NeuralNetwork:
    """
    A class used to represent a single neural network

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

            class __ActivationFunction:
                """
                A class used to represent an activation function

                Attributes:
                    func_type (str): type of activation function
                """

                __slots__ = ['func_type']

                def __init__(self, func_type: str = "step") -> None:
                    available_activation_function_types = ['step', 'linear', 'relu']
                    if func_type.lower() in available_activation_function_types:
                        self.func_type = func_type.lower()
                    else:
                        self.func_type = 'step'

                def forward(self, neuron_value: float) -> float:
                    """
                    Compute a neuron output

                    Args:
                        neuron_value (flat): Weighted inputs + bias

                    Returns:
                        float: A neuron's output
                    """

                    neuron_output = 0.0

                    if self.func_type == 'step':
                        if neuron_value <= 0.0:
                            neuron_output = 0.0
                        else:
                            neuron_output = 1.0
                    elif self.func_type == 'linear':
                        neuron_output = neuron_value
                    elif self.func_type == 'relu':
                        if neuron_value <= 0.0:
                            neuron_output = 0
                        else:
                            neuron_output = neuron_value

                    return neuron_output

            __slots__ = ['weights', 'bias', '__activation_function']

            def __init__(self, no_inputs: int, activation_function_type: str) -> None:
                """
                Creates a single neuron

                Args:
                    no_inputs (int): number of neuron's inputs
                    activation_function_type (str): type of activation function
                """

                self.weights = 1 * np.random.randn(no_inputs)
                self.bias = 0.0
                self.__activation_function = self.__ActivationFunction(activation_function_type)

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

                output = self.__activation_function.forward(output)

                return output

        __slots__ = ['__neurons']

        def __init__(self, no_inputs: int, no_neurons: int, activation_function_type: str) -> None:
            """
            Creates a layer of neurons

            Args:
                no_inputs (list): A list of numbers of neurons inputs
                no_neurons (list): A list of numbers of neurons in layer
                activation_function_type (str): type of activation function
            """

            self.__neurons = []
            for neuron in range(no_neurons):
                self.__neurons.append(self.__Neuron(no_inputs, activation_function_type))

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

    def __init__(self, layer_no_inputs: list, layer_no_neurons: list, layer_activation_functions: list) -> None:
        """
        Creates neural network's layers

        Args:
            layer_no_inputs (list): A list containing numbers of neuron's inputs
            layer_no_neurons (list): A list containing numbers of layer's neurons
            layer_activation_functions (list): A list of activation functions
        """

        self.__layers = []
        for no_inputs, no_neurons, activation_function_type in zip(layer_no_inputs, layer_no_neurons,
                                                                   layer_activation_functions):
            self.__layers.append(self.__Layer(no_inputs, no_neurons, activation_function_type))

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
