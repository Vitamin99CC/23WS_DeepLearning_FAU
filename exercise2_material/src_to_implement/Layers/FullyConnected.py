from Layers.Base import *
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size) # m+1 x n
        self._optimizer = None
        self._gradient_weights = None
        # w11 w12 w1n w1b
        # w21 w22 w2n w2b
        # wm1 wm2 wmn w3b
        # m for input size
        # n for hidden size / output size
        # self.biases = np.zeros(output_size)


    def forward(self, input_tensor):
        if input_tensor.ndim != 2:
            raise Exception("Input tensor has wrong dimension!")
        if input_tensor.shape[1] != self.input_size:
            raise Exception("Input tensor has wrong shape!", input_tensor.shape[1], self.input_size)
        input_tensor = np.append(input_tensor, np.ones((input_tensor.shape[0], 1)), axis=1)
        # b x m+1
        output_tensor = input_tensor.dot(self.weights) # b*m+1 x m+1*n = b x n
        self.input_tensor = input_tensor # b x m+1
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.getter
    def optimizer(self):
        return self._optimizer
    
    def backward(self, error_tensor):
        e_n_minus_1 = error_tensor.dot(self.weights[0:self.input_size, :].T) # ? x n*m+1
        self.gradient_weights = self.input_tensor.T.dot(error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return e_n_minus_1

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_weights.getter
    def gradient_weights(self):
        return self._gradient_weights
    

    def initialize(self, weights_initializer, bias_initializer):
        self.weights[0:self.input_size, :] = weights_initializer.initialize(self.input_size, self.output_size)
        self.weights[self.input_size, :] = bias_initializer.initialize(self.output_size, 1)
        # self.biases = bias_initializer.initialize(self.biases.shape[0], self.biases.shape[1])