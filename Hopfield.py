import numpy as np

from matplotlib import pyplot as plt


class HopfieldNet:
    def __init__(self, numb_neuron):
        """
        :param numb_neuron: number of neurons
        """
        self.numb_neuron = numb_neuron
        self.Neuron = np.zeros((numb_neuron, 1))
        self.weights = np.zeros((numb_neuron, numb_neuron))

    def train(self, trainings_data):
        """

        :param trainings_data: Data-Set to train from
        :return: reconstructed Pattern
        """
        numb_patterns = trainings_data.shape[1]
        # Hebbian learning rule
        for iteration in range(10):
            for i in range(self.numb_neuron):
                for j in range(self.numb_neuron):
                    if i == j:
                        continue

                    bit_sum = 0
                    for pattern in range(numb_patterns):
                        bit_sum += trainings_data[i, pattern] * trainings_data[j, pattern]

                    self.weights[i, j] += (1 / numb_patterns) * bit_sum

    def recpattern(self, pattern, iterations):
        self.Neuron = pattern

        for iteration in range(iterations):
            for i in range(self.numb_neuron):
                input_sum = 0
                for j in range(self.numb_neuron):
                    input_sum += self.weights[i, j] * self.Neuron[j, 0]

                old_neuron_state = self.Neuron
                if input_sum >= -150:
                    self.Neuron[i, 0] = 1
                else:
                    self.Neuron[i, 0] = -1

            if np.all(np.abs(old_neuron_state - self.Neuron)) <= 10**-3:
                print("Minimum reached after", iteration+1, "iteration(s)")
                return self.Neuron

        return self.Neuron
