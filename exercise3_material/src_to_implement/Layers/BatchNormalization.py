import numpy as np
from Layers import Base
from Layers import Helpers
import copy
class BatchNormalization(Base.BaseLayer):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.trainable= True
        self.decay= 0.8
        self.MA_Mean= None
        self.MA_var= None
        self.epsilon= 10**-20
        self.initialize()
        self._optimizer=None
    def initialize(self,weight_initializer=None,bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)
    def forward(self, input_tensor):
        conv= False
        if input_tensor.ndim == 4:
            conv = True
            input_tensor = self.reformat(input_tensor)
        self.input_tensor = input_tensor
        self.mean = np.mean(input_tensor, axis=0)
        self.var = np.var(input_tensor, axis=0)


        if self.testing_phase:
            self.X_hat = (input_tensor - self.MA_Mean) / np.sqrt(self.MA_var + self.epsilon)
        else:
            if self.MA_Mean is None:
                self.MA_Mean = copy.deepcopy(self.mean)
                self.MA_var = copy.deepcopy(self.var)
            else:
                self.MA_Mean = (self.decay * self.MA_Mean) + (1 - self.decay) * self.mean
                self.MA_var = (self.decay * self.MA_var) + (1 - self.decay) * self.var
            self.X_hat = (input_tensor - self.mean) / np.sqrt(self.var + self.epsilon)

        Y_hat = self.gamma * self.X_hat + self.beta

        if conv:
            Y_hat = self.reformat(Y_hat)

        return Y_hat


    def backward(self, error_tensor):
        self.conv = False
        if error_tensor.ndim == 4:
            self.conv= True
            error_tensor = self.reformat(error_tensor)

        d_gamma =np.sum(error_tensor*self.X_hat, axis=0)
        d_beta =np.sum(error_tensor,axis=0)
        if self.testing_phase is False:
            d_X_hat=Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.gamma, self.mean, self.var)
        elif self.testing_phase:
            d_X_hat = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.gamma, self.MA_Mean, self.MA_var)
        if self._optimizer is not None:
            self._optimizer.weight.calculate_update(self.gamma, d_gamma)
            self._optimizer.bias.calculate_update(self.beta, d_beta)


        if self.conv is True:
            d_X_hat = self.reformat(d_X_hat)
        self.gradient_weights = d_gamma
        self.gradient_bias = d_beta
        return d_X_hat


    def reformat(self,tensor):
         if tensor.ndim == 4:
             self.tensor_shape = tensor.shape
             self.B, self.H, self.M, self.N = tensor.shape
             tensor=tensor.reshape(self.B,self.H,self.M * self.N)
             tensor=tensor.transpose(0,2,1)
             tensor=tensor.reshape(self.B* self.M*self.N,self.H)
         else:
             self.B, self.H, self.M, self.N = self.tensor_shape
             tensor = tensor.reshape(self.B, self.M * self.N, self.H)
             tensor = tensor.transpose(0, 2, 1)
             tensor = tensor.reshape(self.B, self.H, self.M, self.N)
         return tensor

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
