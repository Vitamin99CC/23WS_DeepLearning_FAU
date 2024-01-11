import numpy as np
from Layers import Base
class Dropout(Base.BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.probability = probability
    def forward(self, input_tensor):
        if self.testing_phase:
            self.drop=np.ones(input_tensor.shape)
        else:
            self.drop= np.random.choice([1,0],p=[self.probability,1-self.probability],size=input_tensor.shape)
            self.drop= self.drop/self.probability
        return input_tensor * self.drop

    def backward(self, error_tensor):

        return error_tensor*self.drop
