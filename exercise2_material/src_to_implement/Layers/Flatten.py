import numpy as np
class Flatten:
    def __init__(self):
    def forward(self, input_tensor):
        input_tensor=np.reshape(input_tensor)
        return input_tensor
    def backward(self, error_tensor):
        error_tensor=np.reshape(error_tensor)
        return error_tensor


