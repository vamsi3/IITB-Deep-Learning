## AUTHOR: Vamsi Krishna Reddy Satti

##################################################################################
# Loss functions
##################################################################################


import numpy as np


class MSELoss:
    def __init__(self):
        pass

    def forward(self, input, target):
        loss = ((input - target) ** 2).mean()
        return loss

    def backward(self, input, target):
        grad_input = 2 * (input - target) / input.size
        return grad_input

    __call__ = forward


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, input, target):
        input_max = input.max(1, keepdims=True)
        input_exp = np.exp(input - input_max)
        input_exp_sum = input_exp.sum(1, keepdims=True)
        input_softmax = input_exp / input_exp_sum
        loss = input_max + np.log(input_exp_sum) - input  # For numerical stability
        loss = loss[np.arange(target.shape[0], dtype=np.long), target].mean()
        self.cache = input_softmax
        return loss

    def backward(self, input, target):
        input_softmax = self.cache
        delattr(self, 'cache')
        grad_input = input_softmax
        grad_input[np.arange(target.shape[0], dtype=np.long), target] -= 1
        grad_input /= input.shape[0]
        return grad_input

    __call__ = forward
