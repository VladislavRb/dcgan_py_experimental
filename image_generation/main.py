import math
import random
import numpy as np


# utils


def random_bw_minus_plus_1():
    return random.random() * 2 - 1


def randomized_array(n: int):
    return [random_bw_minus_plus_1() for _ in range(n)]


# activation functions

def sigmoid(x: float):
    return 1 / (1 + math.e ** (-x))


def tanh(x: float):
    return (math.e ** x - math.e ** (-x)) / (math.e ** x + math.e ** (-x))


# neurons


class Neuron:
    def __init__(self, init_bias=True):
        self.activation_function = sigmoid
        self.bias = random_bw_minus_plus_1() if init_bias else 0


# layers


class SynapticNet:
    def __init__(self, left_neurons: int, right_neurons: int):
        self.left = [Neuron() for _ in range(left_neurons)]
        self.right = [Neuron() for _ in range(right_neurons)]


class Dense(SynapticNet):
    def __init__(self, left_neurons: int, right_neurons: int):
        super(Dense, self).__init__(left_neurons, right_neurons)
        self.weights = np.array([randomized_array(right_neurons) for _ in range(left_neurons)])


# layers, specific to DCGAN


class BinaryConvKernel:
    def __init__(self, input_size: int):
        # input_size weights to update
        self._matrix = np.zeros((input_size / 4, input_size))

        inner = np.array([randomized_array(2), randomized_array(2)])
        two_d_length = int(input_size ** 0.5) #8
        half_of_two_d_length = two_d_length // 2 # 4

        for i in range(half_of_two_d_length):
            for j in range(half_of_two_d_length):
                row = i * half_of_two_d_length + j
                j_start = i * input_size / 4 + 2 * j

                self._matrix[row, j_start] = inner[0, 0]
                self._matrix[row, j_start + 1] = inner[0, 1]
                self._matrix[row, j_start + two_d_length] = inner[1, 0]
                self._matrix[row, j_start + two_d_length + 1] = inner[1, 1]

    def compute(self, entry: np.ndarray[float]):
        return np.dot(self._matrix, entry)


# model


#class DcganDiscriminator:
#    def __init__(self):


if __name__ == '__main__':
    print(3)
