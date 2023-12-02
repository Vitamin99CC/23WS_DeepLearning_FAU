import numpy as np
from Layers.Base import *

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(input_tensor, 0)
    
    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0)