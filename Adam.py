import numpy as np
from MLP.layer import Affine

class Adam():
    def __init__(self, layers):
        self.first_moment_decay_rate = 0.9
        self.second_moment_decay_rate = 0.999
        self.first_moment_W = dict()
        self.second_moment_W = dict()
        self.first_moment_b = dict()
        self.second_moment_b = dict()
        self.step = 1

        for layer_name, layer_instance in layers.items():
            if hasattr(layer_instance, 'W'):
                self.first_moment_W[layer_name] : np.array = np.zeros_like(layer_instance.W)
                self.second_moment_W[layer_name] : np.array = np.zeros_like(layer_instance.W)
                self.first_moment_b[layer_name] : np.array = np.zeros_like(layer_instance.b)
                self.second_moment_b[layer_name] : np.array = np.zeros_like(layer_instance.b)

    def update(self, learning_rate : float, layer_name : str, layer_instance : Affine):
        EPSILON = 1e-8

        W = layer_instance.W
        b = layer_instance.b
        dW = layer_instance.dW
        db = layer_instance.db


        self.first_moment_W[layer_name] = (self.first_moment_decay_rate * self.first_moment_W[layer_name]) + ((1 - self.first_moment_decay_rate) * dW)
        self.second_moment_W[layer_name] = (self.second_moment_decay_rate * self.second_moment_W[layer_name]) + ((1 - self.second_moment_decay_rate) * (dW * dW))
        self.first_moment_b[layer_name] = (self.first_moment_decay_rate * self.first_moment_b[layer_name]) + ((1 - self.first_moment_decay_rate) * db)
        self.second_moment_b[layer_name] = (self.second_moment_decay_rate * self.second_moment_b[layer_name]) + ((1 - self.second_moment_decay_rate) * (db * db))

        corrected_first_moment_W = (1 / (1 - self.first_moment_decay_rate ** self.step)) * self.first_moment_W[layer_name]
        corrected_first_moment_b = (1 / (1 - self.first_moment_decay_rate ** self.step)) * self.first_moment_b[layer_name]
        corrected_second_moment_W = (1 / (1 - self.second_moment_decay_rate ** self.step)) * self.second_moment_W[layer_name]
        corrected_second_moment_b = (1 / (1 - self.second_moment_decay_rate ** self.step)) * self.second_moment_b[layer_name]

        new_W = W - (learning_rate / (np.sqrt(corrected_second_moment_W) + EPSILON) * corrected_first_moment_W)
        new_b = b - (learning_rate / (np.sqrt(corrected_second_moment_b) + EPSILON) * corrected_first_moment_b)
        return new_W, new_b