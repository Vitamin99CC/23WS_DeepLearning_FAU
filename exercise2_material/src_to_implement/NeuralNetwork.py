from Layers import *
from Optimization import *
from copy import deepcopy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer): 
        self.optimizer = optimizer
        self.wi = weights_initializer
        self.bi = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        hidden_tensor = self.input_tensor
        for layer in self.layers:
            hidden_tensor = layer.forward(hidden_tensor)
        return self.loss_layer.forward(hidden_tensor, self.label_tensor)
        
    
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.wi, self.bi)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()


    def test(self, input_tensor):
        hidden_tensor = input_tensor
        for layer in self.layers:
            hidden_tensor = layer.forward(hidden_tensor)
        return hidden_tensor