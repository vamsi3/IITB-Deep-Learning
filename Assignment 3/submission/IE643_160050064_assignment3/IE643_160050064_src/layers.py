## AUTHOR: Vamsi Krishna Reddy Satti

##################################################################################
# Layers
##################################################################################


import math
import numpy as np


class Layer:
    def __init__(self):
        self.params, self.grad = {}, {}
        self.training = True

    def init_grad(self):
        for name in self.params:
            self.grad[name] = np.zeros_like(self.params[name])
        return self

    def zero_grad(self):
        for name in self.params:
            self.grad[name].fill(0.0)
        return self


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()

        bound = math.sqrt(3 / input_size)
        self.params['weight'] = np.random.uniform(-bound, bound, (input_size, output_size))
        bound = math.sqrt(1 / input_size)
        self.params['bias'] = np.random.uniform(-bound, bound, (1, output_size))
        self.init_grad()

    def forward(self, input):
        output = input @ self.params['weight'] + self.params['bias']
        return output

    def backward(self, input, grad_output):
        grad_input = grad_output @ self.params['weight'].T
        self.grad['weight'] += input.T @ grad_output
        self.grad['bias'] += grad_output.sum(0, keepdims=True)
        return grad_input


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = 1 / (1 + np.exp(-input))
        self.cache = output
        return output

    def backward(self, input, grad_output):
        output = self.cache
        delattr(self, 'cache')
        grad_input = grad_output * output * (1 - output)
        return grad_input


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input_max = input.max(1, keepdims=True)
        output = np.exp(input - input_max)  # For numerical stability
        output /= output.sum(1, keepdims=True)
        self.cache = output
        return output

    def backward(self, input, grad_output):
        output = self.cache
        delattr(self, 'cache')
        grad_input = output * (grad_output - np.sum(grad_output * output, 1, keepdims=True))
        return grad_input
