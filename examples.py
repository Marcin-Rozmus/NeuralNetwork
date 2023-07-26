"""
    Neural network test file.

    Tested neural network structure:
    input layer - 4 neurons
    hidden layer 1 - 3 neurons
    hidden layer 2 - 3 neurons
    output layer - 2 neurons
"""

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
    '''
    Main function. Create neural network and test it.
    '''
    # batch of inputs
    nn_inputs = [[1, 2, 3],
                 [2, 3, 4],
                 [-1, -1, -1.3],
                 [-2.9, -3.2, -4]]

    # hidden layer 1 weights - 3 neurons, 4 inputs
    h_layer1_weights = [[1.0, 0.5, -0.2, 0.1], [1.3, 0.8, 0.5, 1], [1.0, 1.0, 1.0, 1.0]]
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
