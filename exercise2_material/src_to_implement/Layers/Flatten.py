import numpy as np
<<<<<<< HEAD
class Flatten:
    def __init__(self):
    def forward(self, input_tensor):
        input_tensor=np.reshape(input_tensor)
        return input_tensor
    def backward(self, error_tensor):
        error_tensor=np.reshape(error_tensor)
        return error_tensor


=======
from Layers import Base


class Flatten(Base):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        input_tensor = input_tensor.flatten()
        return input_tensor


    def backward(self, error_tensor):
        error_tensor = error_tensor.flatten()
        return error_tensor
>>>>>>> 7e707d6 (Conv Flatten)
