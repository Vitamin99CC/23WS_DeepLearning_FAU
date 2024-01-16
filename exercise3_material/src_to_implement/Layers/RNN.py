import numpy as np
from Layers import Base
from Layers import TanH
from Layers import Sigmoid
from Layers import FullyConnected
class RNN(Base.BaseLayer):
    def __init__(self,input_size, hidden_size, output_size):
        #super().__init__()
        self.trainable = True
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_states = np.zeros(hidden_size) # 1D hidden state
        self.output_tensor = None
        self.hidden_tensor = None
        
        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        #self.whh = np.random.rand(hidden_size, hidden_size)
        #self.wxh = np.random.rand(input_size, hidden_size)
        #self.bh = np.random.rand(1, hidden_size)

        self.fcy = FullyConnected.FullyConnected(hidden_size, output_size)
        self.fch = FullyConnected.FullyConnected(input_size+hidden_size, hidden_size)
        self.sig = Sigmoid.Sigmoid()
        self.tanh = TanH.TanH()

    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self, value):
        self._memorize = value   
    @memorize.getter
    def memorize(self):
        return self._memorize
    
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self.fch.optimizer = value
        self.fcy.optimizer = value
        self._optimizer = value   
    @optimizer.getter
    def optimizer(self):
        return self._optimizer
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    @gradient_weights.getter
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def weights(self):
        return self.fch.weights
    @weights.setter
    def weights(self, value):
        self.fch.weights = value
    @weights.getter
    def weights(self):
        return self.fch.weights

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        self.hidden_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        
        self.tanh_input = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.fch_input = np.zeros((input_tensor.shape[0], self.hidden_size + self.input_size + 1))
        self.fcy_input = np.zeros((input_tensor.shape[0], self.hidden_size + 1))
        if self.memorize == False:
            self.hidden_states = np.zeros(self.hidden_size)

        for i in range(input_tensor.shape[0]):
            #print(self.hidden_states)
            self.tanh_input[i, :] = self.fch.forward(np.concatenate((self.hidden_states, input_tensor[i, :]), axis=0).reshape(1, -1))
            self.fch_input[i, :] = self.fch.input_tensor
            self.hidden_states = self.tanh.forward(self.tanh_input[i, :]).reshape(-1)
            self.hidden_tensor[i, :] = self.hidden_states
        
        self.sig_input = self.fcy.forward(self.hidden_tensor)
        self.fcy_input = self.fcy.input_tensor
        self.output_tensor = self.sig.forward(self.sig_input)

        return self.output_tensor



    def backward(self, error_tensor):
        self.gradient_weights = np.zeros_like(self.fch.weights, dtype=np.float128)
        #print(self.gradient_weights.shape)
        self.y_gradient_weights = np.zeros_like(self.fcy.weights, dtype=np.float128)
        #print(self.y_gradient_weights.shape)
        e_h = 0
        e_n_minus_1 = np.zeros((error_tensor.shape[0], self.input_size))

        #flag_overflow = False

        self.gw = []
        cursor = 0
        for i in range(0, 100):
            self.gw.append(np.zeros_like(self.fch.weights, dtype=np.float128))

        for i in reversed(range(error_tensor.shape[0])):
            #print(error_tensor[i, :].shape)
            self.fcy.input_tensor = self.fcy_input[i, :].reshape(1, -1)
            self.sig.output_tensor = self.output_tensor[i, :].reshape(1, -1)

            # weights = hidden x output = 7 x 5
            # input = hidden = 7
            # error = output = 5
            # gradient = 1x7 5x1
            # input = 3x7
            # output = 3x5 = error
            # gradient = 7x3 3x5 = 7x5
            e_o = self.fcy.backward(self.sig.backward(error_tensor[i, :].reshape(1, -1)))
            # print(e_o.shape) (9, 7)
            self.fch.input_tensor = self.fch_input[i, :].reshape(1, -1)
            self.tanh.output_tensor = self.hidden_tensor[i, :].reshape(1, -1)
            e_xh = self.fch.backward(self.tanh.backward((e_o + e_h).reshape(1, -1)))

            e_h = e_xh[:, 0:self.hidden_size]
            e_x = e_xh[:, self.hidden_size:self.hidden_size+self.input_size]
            e_n_minus_1[i, :] = e_x

            self.y_gradient_weights += self.fcy.gradient_weights

            #if np.max(self.fch.gradient_weights) > 1e+5:
            #    print(self.fch.gradient_weights)
            self.gw[cursor] += self.fch.gradient_weights
            if np.max(self.gw[cursor]) > 1e+15:
                cursor += 1
                #print(cursor)



#            if np.max(self.gradient_weights) > 1e+18:
#                self.gradient_weights2 += self.fch.gradient_weights
#                flag_overflow = True
#            else:
#                self.gradient_weights += self.fch.gradient_weights


        if self._optimizer is not None:
#            if flag_overflow:
#                self.fch.weights = self._optimizer.calculate_update(self.fch.weights, self.gradient_weights2)
            for i in range(0, cursor+1):
                self.fch.weights = self._optimizer.calculate_update(self.fch.weights, self.gw[i])
            self.fch.weights = self._optimizer.calculate_update(self.fch.weights, self.gradient_weights)
            self.fcy.weights = self._optimizer.calculate_update(self.fcy.weights, self.y_gradient_weights)


        return e_n_minus_1



#        e_n_minus_1 = np.zeros((error_tensor.shape[0], self.input_size))
#        e_h = np.zeros((error_tensor.shape[0], self.hidden_size))
#        for i in range(error_tensor.shape[0]):
#            e_o = self.sig.backward(error_tensor[i].dot(self.why.T) + e_h)
#            e_h = e_o.dot(self.why.T) * self.tanh.backward(self.input_tensor[i].dot(self.wxh) + self.hidden_states.dot(self.whh) + self.bh)
#            e_n_minus_1[i, :] = e_h.dot(self.wxh.T)
#        self.gradient_weights = self.input_tensor.T.dot(e_h)
#        if self._optimizer is not None:
#            self.whh = self._optimizer.calculate_update(self.whh, self.gradient_weights)
#        return e_n_minus_1


    def initialize(self, weights_initializer, bias_initializer):
        
        self.fch.initialize(weights_initializer, bias_initializer)
        self.fcy.initialize(weights_initializer, bias_initializer)

        #self.whh[0:self.hidden_size, :] = weights_initializer.initialize([self.hidden_size, self.hidden_size], self.self.hidden_size, self.self.hidden_size)
        #self.whh[self.hidden_size, :] = bias_initializer.initialize([1, self.hidden_size], self.hidden_size, 1)

        #self.wxh[0:self.input_size, :] = weights_initializer.initialize([self.input_size, self.hidden_size], self.self.input_size, self.self.hidden_size)
        #self.wxh[self.input_size, :] = bias_initializer.initialize([1, self.hidden_size], self.hidden_size, 1)

        #self.why[0:self.hidden_size, :] = weights_initializer.initialize([self.hidden_size, self.output_size], self.self.hidden_size, self.self.output_size)
        #self.why[self.hidden_size, :] = bias_initializer.initialize([1, self.output_size], self.output_size, 1)