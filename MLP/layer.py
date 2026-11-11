import numpy as np
import MLP.activation_function as af

class Node:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass

class Affine(Node):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x : np.array):
        self.x = x.copy()
        return af.Affine(self.x, self.W, self.b)

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx

class ReLU(Node):
    def __init__(self):
        self.mask = None

    def forward(self, x : np.array):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
        
class SoftmaxWithLoss(Node):
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x : np.array, t : np.array, TD_error : float):
        self.y = af.softmax(x)
        self.t = t
        self.loss = af.cross_entropy_error(self.y, self.t)
        self.TD_error = TD_error
        return TD_error * self.loss

    def backward(self, entropy_coefficient = 0.01, dout=1.0):
        batch_size = self.y.shape[0]
        entropy = -np.sum(self.y * np.log(self.y + 1e-7), axis=1, keepdims=True)
        out = (self.TD_error * dout * (self.y - self.t) + entropy_coefficient * self.y * (np.log(self.y + 1e-7) + entropy)) / batch_size
        return out

class MeanSquaredError(Node):
    def __init__(self):
        self.y = None
        self.t = None
        self.size = None

    def forward(self, x, t):
        self.y = x.copy()
        self.t = t.copy()
        self.size = x.shape[0]
        return af.mean_squared_error(self.y, self.t)

    def backward(self, dout=1):
        return dout / self.size * 2 * (self.y - self.t)