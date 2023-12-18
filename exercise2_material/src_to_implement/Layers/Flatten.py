import numpy as np
from Layers.Base import *

import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        # Save the input shape for later use in backward
        self.input_shape = input_tensor.shape

        # Reshape the input tensor to a one-dimensional feature vector
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        # Reshape the error tensor back to the original shape
        return error_tensor.reshape(self.input_shape)
