from Layers.Base import *
import numpy as np
import math

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.gradient = None
        self.input_tensor = None
        self.input_shape = None
        self.output_shape = None

        if type(self.stride_shape) != tuple:
            self.stride_shape = (self.stride_shape, self.stride_shape)
            

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape
        batch_size, channels, height, width = self.input_shape
        self.output_shape = (math.floor((self.input_shape[0] - self.pooling_shape[0]) / self.stride_shape[0]) + 1, 
                             math.floor((self.input_shape[1] - self.pooling_shape[1]) / self.stride_shape[1]) + 1)
                
        input_reshaped = input_tensor.reshape(batch_size, channels, self.output_shape[0], 
                                              self.pooling_shape[0], self.output_shape[1], 
                                              self.pooling_shape[1])

        pooled = np.max(input_reshaped, axis=(-1, -3))

        self.arg_max = np.argmax(input_reshaped, axis=(-1, -3))
        return pooled
    

    def backward(self, grad_output):
        batch_size, channels, height, width = self.input.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride

        grad_input = np.zeros_like(self.input)

        # Compute gradients using stored indices
        for b in range(batch_size):
            for c in range(channels):
                for i in range(height):
                    for j in range(width):
                        idx_h = i * sh
                        idx_w = j * sw
                        idx_max = self.arg_max[b, c, i, j]
                        h_max, w_max = divmod(idx_max, kw)
                        grad_input[b, c, idx_h + h_max, idx_w + w_max] += grad_output[b, c, i, j]

        return grad_input