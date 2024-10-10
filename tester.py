"""
    Neural network test file.

"""

import Neuron


def main():
    """
        Main function. Create neural network and test it.

    """

    inputs = [1.0, 2.0, 3.0]

    neuron = Neuron.Neuron(3)

    print(neuron.forward(inputs))


if __name__ == '__main__':
    main()
