import numpy as np
from MLP.neural_network import NeuralNetwork
import MLP.activation_function as af
import MLP.layer as ly

class PolicyNeuralNetwork(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self._init_network()
        self.learning_rate = 0.098

    def _init_network(self):
        '''
        input data size = 8
        input data form(observation) = [x_position, y_position, x_velocity, y_velocity, lander_angle, 
        angular_velocity, left_leg_contact (boolean), right_leg_contact (boolean)]
        '''

        # Input Layer
        W1 = np.random.randn(8, 128) * np.sqrt(2 / 8) # He initial value
        b1= np.zeros(128)
        # Hidden Layer 1
        W2 = np.random.randn(128, 64) * np.sqrt(2 / 128) # He initial value
        b2 = np.zeros(64)
        # Hidden Layer 2
        W3 = np.random.randn(64, 4) * np.sqrt(2 / 64) # He initial value
        b3 = np.zeros(4)
        
        self.layers['Affine1'] = ly.Affine(W1, b1)
        self.layers['ReLU1'] = ly.ReLU()
        self.layers['Affine2'] = ly.Affine(W2, b2)
        self.layers['ReLU2'] = ly.ReLU()
        self.layers['Affine3'] = ly.Affine(W3, b3)
        self.last_layer = ly.SoftmaxWithLoss()

    def predict(self, x : np.array):
        return super().predict(x)

    def _loss(self, x, t, td_error):
        y = self.predict(x)
        return self.last_layer.forward(y, t, td_error)

    def gradient(self, x, t, td_error):
        self.loss(x, t, td_error)
        dout = self.last_layer.backward()
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

    def get_actions_probabilities(self, state):
        return self.predict(state)

    def get_policy_log_probability_gradient(self, action, state):
        pass
    
    def gradient_ascent(self, objective_function_gradient):
        pass # adam