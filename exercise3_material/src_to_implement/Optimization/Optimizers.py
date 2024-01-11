import numpy as np

class Optimizers(object):
    def __init__(self):
        self.regularizer = None
    def add_regularizer(self, regularizer):
        self.regularizer =regularizer
class Sgd(Optimizers):
    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is None :
         weight_tensor -= self.lr * gradient_tensor
        else:
         weight_tensor -= (self.lr*self.regularizer.calculate_gradient(weight_tensor)) - (self.lr * gradient_tensor)
        return weight_tensor


class SgdWithMomentum(Optimizers):
    def __init__(self,learning_rate,momentum_rate):
        super().__init__()
        self.lr = learning_rate
        self.momentum_rate =momentum_rate
        self.velocity=None
    def calculate_update(self,weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity =np.zeros_like(weight_tensor)
        self.velocity = (self.momentum_rate * self.velocity)-(self.lr * gradient_tensor)
        if self.regularizer is None:
            return weight_tensor+self.velocity
        else:
            return weight_tensor+self.velocity-self.regularizer.calculate_gradient(weight_tensor)


class Adam(Optimizers):
    def __init__(self, learning_rate, mu ,rho):
        super().__init__()
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
        if self.regularizer is None:
            return weight_tensor- self.lr*(self.v_hat/(np.sqrt(self.r_hat)+1e-8))
        else:
            return weight_tensor- self.regularizer.calculate_gradient(weight_tensor)-self.lr*(self.v_hat/(np.sqrt(self.r_hat)+1e-8))

''''import numpy as np


class Optimizer(object):
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if type(weight_tensor) is not np.ndarray:
            temp = weight_tensor
        else:
            temp = weight_tensor.copy()
        weight_tensor -= self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(temp)
        return weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0.

    def calculate_update(self, weight_tensor, gradient_tensor):
        if type(weight_tensor) is not np.ndarray:
            temp = weight_tensor
        else:
            temp = weight_tensor.copy()

        v = self.learning_rate * gradient_tensor + self.momentum_rate * self.v
        weight_tensor = weight_tensor - v
        self.v = v
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(temp)
        return weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0.
        self.r = 0.
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        if type(weight_tensor) is not np.ndarray:
            temp = weight_tensor
        else:
            temp = weight_tensor.copy()
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.power(gradient_tensor, 2)
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        self.k += 1
        weight_tensor = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(temp)
        return weight_tensor '''''




