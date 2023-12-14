import numpy as np
from Layers.Base import *


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        input_tensor = input_tensor.flatten()
        return input_tensor


    def backward(self, error_tensor):
        error_tensor = error_tensor.flatten()
        return error_tensor
