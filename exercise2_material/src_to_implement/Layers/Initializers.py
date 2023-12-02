import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value
    
    def initialize(self, weights_shape, fan_in, fan_out):



class UniformRandom:
    
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weights = np.random.uniform(-1, 1, weights_shape)


class Xavier:




class He: 