import numpy as np
from activation import unit_step, sigmoid

class Perceptron:
    def __init__(self, inputs, activation_function=sigmoid, bias=1.0):
        self.weights = np.random.rand(inputs)
        self.bias = bias
        self.activation_function = activation_function

    def set_weights(self,w_init):
        self.weights = np.array(w_init)

    def set_bias(self,b_init):
        self.bias = np.array(b_init)

    def run(self, x):
        x_sum = np.dot(x, self.weights) + self.bias
        return self.activation_function(x_sum)
