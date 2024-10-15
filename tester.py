"""
    Neural network test file.

"""

import Neuron
import Layer


def main():
    """
    Main function. Create neural network and test it.

    """

    inputs = [1.0, 2.0, 3.0, -1.0]

    layer = Layer.Layer(no_inputs=4, no_neurons=3)

    print(layer.forward(inputs))


if __name__ == "__main__":
    main()
