import numpy as np


class Neuron:
    """
    A class used to represent a single neuron

    Args:
        no_inputs (int): A number of neurons inputs. This argument determine number of weights.

    """

    __slots__ = ["__weights", "__bias"]

    def __init__(self, no_inputs: int, weights=(), bias=0.0):
        if len(weights) == no_inputs:
            self.__weights = list(weights)
        else:
            self.__weights = np.random.randn(no_inputs)
        self.__bias = bias

    def forward(self, inputs: list) -> float:
        """
        Computes a neuron's output

        Args:
            inputs (list): A list of neuron's inputs

        Returns:
            float: Calculated neuron's output
        """
        output = np.dot(self.__weights, inputs) + self.__bias

        return output
