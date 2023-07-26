"""
    Neural network test file.

    Tested neural network structure:
    input layer - 2 neurons
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

    # hidden layer 1 - 3 neurons, 2 inputs
    h_layer1_no_inputs = 2
    h_layer1_no_neurons = 3
    # hidden layer 2 - 3 neurons, 3 inputs
    h_layer2_no_inputs = 3
    h_layer2_no_neurons = 3
    # output layer - 2 neurons, 3 inputs
    o_layer_no_inputs = 3
    o_layer_no_neurons = 2

    nn_no_inputs = [h_layer1_no_inputs, h_layer2_no_inputs, o_layer_no_inputs]
    nn_no_neurons = [h_layer1_no_neurons, h_layer2_no_neurons, o_layer_no_neurons]

    # neural network creation
    nn = NeuralNetwork.NeuralNetwork(nn_no_inputs, nn_no_neurons)
    # compute neural network output
    nn_outputs = nn.forward(nn_inputs)

    # print results
    print_nn_outputs(nn_outputs)


if __name__ == '__main__':
    main()
