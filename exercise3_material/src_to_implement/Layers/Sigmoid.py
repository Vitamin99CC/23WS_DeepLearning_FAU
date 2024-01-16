import numpy as np
from Layers import Base
class Sigmoid(Base.BaseLayer):

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if np.min(input_tensor) < -709:
            self.output_tensor = np.exp(input_tensor) / (1 + np.exp(input_tensor))
        else:
            self.output_tensor = 1 / (1 + np.exp(-input_tensor))
        
        return self.output_tensor

    def backward(self, error_tensor):
        # temp = self.forward(error_tensor
        
        return self.output_tensor * (1 - self.output_tensor) * error_tensor