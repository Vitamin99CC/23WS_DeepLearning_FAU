import numpy as np
from Layers import Base
class RNN(Base.BaseLayer):
    def __init__(self,input_size,hidden_size,output_size):
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =output_size
        self.hidden_states= np.zeros(hidden_size)
        self.memorize = False
    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self,value):
        self._memorize = value   

    def forward(self,input_tensor):




    def backward(self,error_tensor):
