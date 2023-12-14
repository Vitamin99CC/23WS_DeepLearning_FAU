import numpy as np
from Layers.Base import *
from Layers.Initializers import *

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.input_tensor = None
        self.output_tensor = None
        self.error_tensor = None
        self.weights_gradient = None
        self.bias_gradient = None
        self.weights_initializer = UniformRandom()
        self.bias_initializer = UniformRandom()
        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.weights = self.weights_initializer.initialize(self.weights_shape, self.fan_in, self.fan_out)
        self.bias = self.bias_initializer.initialize(self.bias_shape, self.fan_in, self.fan_out)

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.getter
    def optimizer(self):
        return self._optimizer