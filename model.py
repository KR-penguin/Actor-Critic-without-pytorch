from poilcy_neural_network import PolicyNeuralNetwork
import numpy as np

class Agent:
    def __init__(self):
        self.discount_factor = 0.98
        self.last_state = None
        self.last_action = None
        self.policy_neural_network = PolicyNeuralNetwork()
        self.approximated_state_value_function_neural_network = None

    def _get_state_value(self, state):
        self.approximated_state_value_function_neural_network.get_approximated_state_value_function_value(state)

    def get_action_from_policy(self, state):
        actions_probabilities = self.policy_neural_network.get_actions_probabilities(state)
        '''
        example output = [ 0.33248537  0.01559577  1.29106821 -0.60009708]
        
        => [NO_OPERATION, EXCUTION_LEFT_ENGINE, EXCUTION_MAIN_ENGINE, EXCUTION_RIGHT_ENGINE]
        '''
        action = np.argmax(actions_probabilities)
        last_action = action
        return action

    def update(self, batch):
        _update_policy(batch)
        _update_approximated_state_value_function(batch)

    def _update_policy(self, TD_error):
        total = 0
        for step in range(batch.batch_size):
            TD_error = batch.rewards[step] + discount_factor * _get_state_value(batch.states[step+1]) - _get_state_value(batch.states[step])
            gradient = policy_neural_network.get_policy_log_probability_gradient(batch.actions[step], batch.states[step])
            total += gradient * TD_error
        objective_function_gradient = total / batch.batch_size
        self.policy_neural_network.gradient_ascent(objective_function_gradient)

    def _update_approximated_state_value_function(self, batch):
        pass
        # self.approximated_state_value_function_neural_network.update_approximated_state_value_function(TD_target, last_state_value)