"""
    Neural network test file.

"""

import numpy as np
import matplotlib.pyplot as plt

import Neuron
import Layer
import Spiral


def main():
    """
    Main function. Create neural network and test it.

    """
    inputs = [1.0, 2.0]
    X, y = Spiral.generate_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    # plt.show()

    layer = Layer.Layer(no_inputs=2, no_neurons=3)
    ys_pred = layer.forward(X[:5])

    for y_pred in ys_pred:
        print(y_pred)


if __name__ == "__main__":
    main()
