import numpy as np
class Conv:
    def __init__(self,stride_shape,convolution_shape,num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels =num_kernels
        self.trainable=True
        self.weights=np.random.rand(num_kernels,*convolution_shape)
        self.bias = np.random.rand(num_kernels)
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias =None

    def forward(self, input_tensor):

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.getter
    def optimizer(self):
        return self._optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape)
        self.bias = bias_initializer.initialize(self.bias.shape)

    def backward(self,output_tensor):


    def initialize(self,weights_initializer, bias_initializer):
