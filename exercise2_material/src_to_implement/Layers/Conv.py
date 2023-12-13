import numpy as np
<<<<<<< HEAD
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
=======
from Layers import Base
from Layers import Initializers

class Conv(Base):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.padding_type = padding_type
        self.weights = None
        self.bias = None
        self.input_tensor_shape = None
        self.input_tensor = None
        self.output_tensor = None
        self.error_tensor = None
        self.error_tensor_shape = None
        self.weights_gradient = None
        self.bias_gradient = None
        self.weights_initializer = Initializers.UniformRandom()
        self.bias_initializer = Initializers.UniformRandom()
        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.weights = self.weights_initializer.initialize(self.weights_shape, self.fan_in, self.fan_out)
        self.bias = self.bias_initializer.initialize(self.bias_shape, self.fan_in, self.fan_out)

    def forward(self, input_tensor):
        

    def backward(self, error_tensor):



>>>>>>> 7e707d6 (Conv Flatten)

    @property
    def optimizer(self):
        return self._optimizer
<<<<<<< HEAD

=======
    
>>>>>>> 7e707d6 (Conv Flatten)
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.getter
    def optimizer(self):
<<<<<<< HEAD
        return self._optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape)
        self.bias = bias_initializer.initialize(self.bias.shape)

    def backward(self,output_tensor):


    def initialize(self,weights_initializer, bias_initializer):
=======
        return self._optimizer
>>>>>>> 7e707d6 (Conv Flatten)
