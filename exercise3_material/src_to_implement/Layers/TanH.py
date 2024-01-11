import numpy as np
from Layers import Base
class TanH(Base.BaseLayer):
    def forward(self, input_tensor):
        self.activ = np.tanh(input_tensor)
        return self.activ

    def backward(self, error_tensor):
        return (1 - np.square(self.activ))*error_tensor