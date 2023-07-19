import Neuron

inputs = [1, 2, 3]

neuron = Neuron.Neuron([1.0, 0.5, -0.2], 1.5)
neuron_output = neuron.forward(inputs)

print(neuron_output)
