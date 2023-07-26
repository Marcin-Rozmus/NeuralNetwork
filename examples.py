"""
    Neural network test file.

    Tested neural network structure:
    input layer - 4 neurons
    hidden layer 1 - 3 neurons
    hidden layer 2 - 3 neurons
    output layer - 2 neurons
"""
import matplotlib.pyplot as plt

import Spiral
import NeuralNetwork


def print_nn_outputs(nn_outputs: list) -> None:
    """
    Print neural network outputs in formatted way

    Args:
        nn_outputs (list): A batch of neural network's outputs
    """
    inputs_no = 1
    for nn_output in nn_outputs:
        print(f'Inputs set {inputs_no}: [', end='')
        output_no = 1
        for neuron_output in nn_output:
            if output_no > 1:
                print(', ', end='')
            print(f'{neuron_output:.2f}', end='')
            output_no += 1
        print(']')
        inputs_no += 1


def main() -> None:
    """
    Main function. Create neural network and test it.

    """

    # generate training data
    X, y = Spiral.generate_data(samples=500, classes=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired')
    plt.title('Training data')
    plt.show()

    # batch of inputs
    nn_inputs = zip(X[:, 0], X[:, 1])

    # hidden layer 1 weights - 3 neurons, 2 inputs
    h_layer1_weights = [[1.0, 0.5], [1.3, 0.8], [1.0, 1.0]]
    # hidden layer 2 weights - 3 neurons, 3 inputs
    h_layer2_weights = [[3.0, 1.2, 1.3], [-0.1, -0.2, -0.3], [1.0, 1.0, 1.0]]
    # output layer weights - 2 neurons, 3 inputs
    o_layer_weights = [[1.0, 1.4, -1.0], [-1.0, -1.5, 2.0]]

    # hidden layer 1 biases - 3 neurons
    h_layer1_biases = [1.5, -2.0, 1.0]
    # hidden layer 2 biases - 3 neurons
    h_layer2_biases = [-1.5, 2.0, -1.0]
    # output layer biases - 2 neurons
    o_layer_biases = [1.0, -1.0]

    neurons_weights = [h_layer1_weights, h_layer2_weights, o_layer_weights]
    neurons_biases = [h_layer1_biases, h_layer2_biases, o_layer_biases]

    # neural network creation
    nn = NeuralNetwork.NeuralNetwork(neurons_weights, neurons_biases)
    # compute neural network output
    nn_outputs = nn.forward(nn_inputs)

    # print results
    print_nn_outputs(nn_outputs)


if __name__ == '__main__':
    main()
