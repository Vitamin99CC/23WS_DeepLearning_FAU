import numpy as np
from Layers.Base import *
from Layers.Initializers import *
from scipy.signal import correlate2d

'''
class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape 
        self.num_kernels = num_kernels
        # c, m, n (c for channel, m n for kernel shape)

        if len(convolution_shape) == 2:  # 1D convolution
            self.weights = np.random.rand(num_kernels, convolution_shape[0])
        elif len(convolution_shape) == 3:  # 2D convolution
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1])
        else:
            raise ValueError("Invalid convolution shape")
        
        self.weights_initializer = UniformRandom()
        self.bias_initializer = UniformRandom()
        self.weights = self.weights_initializer.initialize(self.weights_shape, self.fan_in, self.fan_out)
        self.bias = self.bias_initializer.initialize(self.bias_shape, self.fan_in, self.fan_out)
        
        self.weights_gradient = None
        self.bias_gradient = None

        self.input_tensor = None
        self.output_tensor = None
        self.error_tensor = None
        self._optimizer = None

        self.output_shape = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape
        
        # 1d: batch, channel, y (spatial dimension)
        # 2d: batch, channel, y, x
        if self.stride_shape == 1:
            x_padding = self.convolution_shape[1] // 2
            y_padding = self.convolution_shape[2] // 2
            self.output_shape = self.input_shape
            self.output_shape = (self.input_shape[0], self.input_shape[2] - self.convolution_shape[1], self.input_shape[3])
        else:
            self.output_shape = self.
            x_padding = 0
            y_padding = 0

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
    


'''

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True

        # Initialize weights and biases
        if len(convolution_shape) == 2:  # 1D convolution
            self.weights = np.random.rand(num_kernels, convolution_shape[0])
        elif len(convolution_shape) == 3:  # 2D convolution
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1])
        else:
            raise ValueError("Invalid convolution shape")

        self.bias = np.zeros(num_kernels)

        # Gradient placeholders
        self._gradient_weights = None
        self._gradient_bias = None

        # Optimizer
        self._optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    def forward(self, input_tensor):
        batch_size, input_channels, input_height, input_width = input_tensor.shape
        num_output_channels = self.num_kernels
        stride_y, stride_x = (
            self.stride_shape if isinstance(self.stride_shape, tuple) else (self.stride_shape, self.stride_shape)
        )

        if len(self.convolution_shape) == 2:  # 1D convolution
            _, kernel_size = self.convolution_shape
            self.weights = self.weights[:, :kernel_size]  # Adjust weights if necessary
            output_height = input_height
            output_width = (input_width - kernel_size) // stride_x + 1

        elif len(self.convolution_shape) == 3:  # 2D convolution
            _, kernel_height, kernel_width = self.convolution_shape
            self.weights = self.weights[:, :kernel_height, :kernel_width]  # Adjust weights if necessary
            output_height = (input_height - kernel_height) // stride_y + 1
            output_width = (input_width - kernel_width) // stride_x + 1

        # Perform convolution
        output_tensor = np.zeros((batch_size, num_output_channels, output_height, output_width))

        for b in range(batch_size):
            for c_out in range(num_output_channels):
                for c_in in range(input_channels):
                    output_tensor[b, c_out] += correlate2d(
                        input_tensor[b, c_in],
                        self.weights[c_out, c_in],
                        mode='same',
                        boundary='fill',
                        fillvalue=0,
                    )

                output_tensor[b, c_out] += self.bias[c_out]

        return output_tensor

    def backward(self, error_tensor):
        batch_size, num_output_channels, output_height, output_width = error_tensor.shape
        input_channels, input_height, input_width = self.weights.shape[1], *self.input_shape[2:]
        stride_y, stride_x = (
            self.stride_shape if isinstance(self.stride_shape, tuple) else (self.stride_shape, self.stride_shape)
        )

        # Initialize gradient placeholders
        gradient_weights = np.zeros_like(self.weights)
        gradient_bias = np.zeros_like(self.bias)
        error_tensor_padded = np.pad(error_tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

        # Backward pass
        for b in range(batch_size):
            for c_out in range(num_output_channels):
                for c_in in range(input_channels):
                    gradient_weights[c_out, c_in] += correlate2d(
                        self.input[b, c_in],
                        error_tensor_padded[b, c_out],
                        mode='valid',
                    )

                gradient_bias[c_out] += np.sum(error_tensor[b, c_out])

        # Update gradients
        self._gradient_weights = gradient_weights
        self._gradient_bias = gradient_bias

        # Update weights using the optimizer
        if self._optimizer:
            self.weights = self._optimizer.update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.update(self.bias, self._gradient_bias)

        # Compute error tensor for the next layer
        error_tensor_next = np.zeros((batch_size, input_channels, input_height, input_width))

        for b in range(batch_size):
            for c_out in range(num_output_channels):
                for c_in in range(input_channels):
                    error_tensor_next[b, c_in] += correlate2d(
                        error_tensor_padded[b, c_out],
                        np.rot90(self.weights[c_out, c_in], 2),
                        mode='full',
                    )

        return error_tensor_next

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.weights = self.weights_initializer.initialize(self.weights_shape, self.fan_in, self.fan_out)
        self.bias = self.bias_initializer.initialize(self.bias_shape, self.fan_in, self.fan_out)
