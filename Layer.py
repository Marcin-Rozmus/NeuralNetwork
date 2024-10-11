import numpy as np


class Layer:
    """
    A class used to represent a singe layer.

    Args:
        no_inputs (int): A number of neurons inputs. This argument determine number of neurons weights.
        no_neurons (int): A number of neurons in layer.
    """

    __slots__ = ["__neurons"]

    def __init__(self, no_inputs: int, no_neurons: int):
        self.__neurons = []
        for i in range(no_neurons):
            neuron = self.__Neuron(no_inputs)
            self.__neurons.append(neuron)

    class __Neuron:
        """
        A class used to represent a single neuron

        Args:
            no_inputs (int): A number of neurons inputs. This argument determine number of weights.

        """

        __slots__ = ["__weights", "__bias"]

        def __init__(self, no_inputs: int):
            self.__weights = np.random.randn(no_inputs)
            self.__bias = 0.0

        def forward(self, inputs: list):
            output = 0
            for i in range(len(inputs)):
                output += self.__weights[i] * inputs[i]
            output += self.__bias

            return output
