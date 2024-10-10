import numpy as np


class Neuron:
    """
    A class used to represent a single neuron

    Args:
        no_inputs (int): A number of neurons inputs. This argument determine number of weights.
        
    """
    
    __slots__ = [
        '__weights',
        '__bias'
    ]

    def __init__(self, no_inputs: int):
        self.__weights = np.random.randn(no_inputs)
        self.__bias = 0.0

    def forward(self, inputs: list):
        output = 0
        for i in range(len(inputs)):
            output += self.__weights[i] * inputs[i]
        output += self.__bias

        return output
