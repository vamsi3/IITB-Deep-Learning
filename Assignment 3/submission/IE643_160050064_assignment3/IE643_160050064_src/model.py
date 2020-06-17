## AUTHOR: Vamsi Krishna Reddy Satti

##################################################################################
# Model
##################################################################################


import numpy as np


class Model:
    def __init__(self, training=True):
        self.layers = []
        self.training = training

    def add_layer(self, layer):
        layer.training = self.training
        self.layers.append(layer)
        return self

    def forward(self, input):
        activations = [input]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        if self.training:
            self.cache = activations  # Cache between forward and backward passes
        return activations[-1]

    def backward(self, grad_output):
        activations = self.cache
        num_layers = len(self.layers)
        for layer_index in range(num_layers - 1, -1, -1):
            grad_output = self.layers[layer_index].backward(activations[layer_index], grad_output)
        delattr(self, 'cache')
        return grad_output

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
        return self

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.training = self.training
        return self

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.training = self.training
        return self

    __call__ = forward
