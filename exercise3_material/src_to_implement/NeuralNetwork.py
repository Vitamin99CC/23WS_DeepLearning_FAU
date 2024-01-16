from Layers import *
from Optimization import *
from copy import deepcopy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer): 
        self._phase = None
        self.optimizer = optimizer
        self.wi = weights_initializer
        self.bi = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        
    @property
    def phase(self):

        return self._phase
    @phase.setter
    def phase(self, value):
        self._phase = value


    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        hidden_tensor = self.input_tensor
        self.reg_loss = 0
        for layer in self.layers:
            hidden_tensor = layer.forward(hidden_tensor)
            if layer.trainable:
                if self.optimizer.regularizer is not None:
                    self.reg_loss += self.optimizer.regularizer.norm(layer.weights)
        return self.loss_layer.forward(hidden_tensor, self.label_tensor) + self.reg_loss
        
    
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
        self.phase = "train"
        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                layer.testing_phase = False
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()


    def test(self, input_tensor):
        self.phase = "test"
        hidden_tensor = input_tensor
        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                layer.testing_phase = True
            hidden_tensor = layer.forward(hidden_tensor)
        return hidden_tensor