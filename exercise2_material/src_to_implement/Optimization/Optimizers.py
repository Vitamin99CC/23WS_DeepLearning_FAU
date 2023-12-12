
class Sgd:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.lr * gradient_tensor
        return weight_tensor

            