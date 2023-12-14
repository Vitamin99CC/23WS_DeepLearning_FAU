import numpy as np
class Sgd:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.lr * gradient_tensor
        return weight_tensor

class SgdWithMomentum:
    def __init__(self,learning_rate,momentum_rate):
        self.lr = learning_rate
        self.momentum_rate =momentum_rate
        self.velocity=None
    def calculate_update(self,weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity =np.zeros_like(weight_tensor)
        self.velocity = (self.momentum_rate * self.velocity)-(self.lr * gradient_tensor)

        return weight_tensor+self.velocity


class Adam:
    def __init__(self, learning_rate, mu ,rho):
        self.lr = learning_rate
        self.mu = mu
        self.rho = rho
        self.momentum_one=None
        self.momentum_two=None
        self.k=0
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k+=1
        if self.momentum_one is None:
            self.momentum_one =np.zeros_like(weight_tensor)
            self.momentum_two=np.zeros_like(weight_tensor)
        self.momentum_one=(self.mu*self.momentum_one)+((1-self.mu)*gradient_tensor)
        self.momentum_two=(self.rho*self.momentum_two)+(1-self.rho)*gradient_tensor**2
        self.v_hat=self.momentum_one/(1-self.mu**self.k)
        self.r_hat=self.momentum_two/(1-self.rho**self.k)
        return weight_tensor- self.lr*(self.v_hat/(np.sqrt(self.r_hat)+1e-8))





