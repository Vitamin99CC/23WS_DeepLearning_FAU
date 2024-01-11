import numpy as np
from Layers.Base import *

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor - np.max(input_tensor)
        #print(input_tensor)
        
        self.yhat = np.exp(self.input_tensor) / np.sum(np.exp(self.input_tensor), axis=1, keepdims=True)
        #print(self.yhat)
        return self.yhat

    def backward(self, error_tensor):
        return self.yhat * (error_tensor - np.sum(self.yhat * error_tensor, axis=1, keepdims=True))
        # potential dimension problem for sum En,j * yhatj


        # -100 -100 0 -100  