import numpy as np

import Neuron


class Layer:
    """
    A class used to represent a singe layer.

    Args:
        no_inputs (int): A number of neurons inputs. This argument determine number of neurons weights.
        no_neurons (int): A number of neurons in layer.
    """

    __slots__ = ["__neurons"]

    def __init__(self, no_inputs: int, no_neurons: int, weights=(), biases=()):
        self.__neurons = []
        for i in range(no_neurons):
            if len(weights) == no_neurons:
                neuron_weigths = weights[i]
            else:
                neuron_weigths = ()
            if len(biases) == no_neurons:
                neuron_bias = biases[i]
            else:
                neuron_bias = None

            if neuron_bias is None:
                neuron = Neuron.Neuron(no_inputs=no_inputs, weights=neuron_weigths)
            else:
                neuron = Neuron.Neuron(
                    no_inputs=no_inputs, weights=neuron_weigths, bias=neuron_bias
                )
            self.__neurons.append(neuron)

    def forward(self, inputs: list) -> list:
        """
        Computes a list of layer's outputs

        Args:
            inputs (list): A batch of layer's inputs

        Returns:
            list: A batch of layer's outputs
        """
        output = []
        for neuron in self.__neurons:
            output.append(neuron.forward(inputs))

        return output
