import numpy as np
import MLP.activation_function as af
import MLP.layer as ly

class NeuralNetwork:
    def __init__(self):
        self.layers = {}
        self.last_layer = None
        self.optimizer = None

    def init_network(self):
        pass
    
    def _predict(self, x : np.array):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def _loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def _gradient(self, x, t):
        self._loss(x, t)
        dout = self.last_layer.backward()
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

    def get_parameters(self):
        parameters = {}
        for name, layer in self.layers.items():
            if hasattr(layer, 'W'):
                parameters[name] = {
                    'W': layer.W.tolist(),
                    'b': layer.b.tolist(),
                }
        return parameters