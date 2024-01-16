import numpy as np
from Layers import Base
class TanH(Base.BaseLayer):
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.tanh(input_tensor)
        return self.output_tensor

    def backward(self, error_tensor):
        return (1 - np.square(self.output_tensor))*error_tensor