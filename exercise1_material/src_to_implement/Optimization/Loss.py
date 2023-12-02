import numpy as np
from copy import deepcopy

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None
        self.trainable = False

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        self.yhat = np.sum(-label_tensor * np.log(prediction_tensor + np.finfo(np.float64).eps))
        return deepcopy(self.yhat)

    def backward(self, label_tensor):
        return -label_tensor / (self.prediction_tensor + np.finfo(np.float64).eps)
    

# dlnx/dx = 1/x
# labels = 0 or 1
# network: reach 0 ASAP
# MSE: linear function, not as fast enough as CrossELoss