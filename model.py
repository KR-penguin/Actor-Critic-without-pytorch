from poilcy_neural_network import PolicyNeuralNetwork
from state_value_neural_network import StateValueNeuralNetwork
import numpy as np

class Agent:
    def __init__(self):
        self.discount_factor = 0.98
        self.last_state = None
        self.policy_neural_network = PolicyNeuralNetwork()
        self.state_value_neural_network = StateValueNeuralNetwork()

    def _get_state_value(self, states : np.array):
        return self.state_value_neural_network.get_approximated_state_value(states)

    def get_action_from_policy(self, state):
        state = np.array(state).reshape(1, -1)
        actions_probabilities = self.policy_neural_network.get_actions_probabilities(state)
        '''
        [[0.4, 0.3, 0.2, 0.1]]

        => [[NO_OPERATION, EXCUTION_LEFT_ENGINE, EXCUTION_MAIN_ENGINE, EXCUTION_RIGHT_ENGINE]]
        '''
        actions_probabilities = actions_probabilities[0]
        actions = np.random.choice(len(actions_probabilities), p=actions_probabilities)
        return actions

    def update(self, batch):
        actions = np.eye(4)[np.array(batch.actions)]
        states = np.array(batch.states)
        next_states = np.array(batch.next_states)
        is_game_over = np.array(batch.is_game_over).reshape(-1, 1)
        rewards = np.array(batch.rewards).reshape(-1, 1)

        next_states_value = np.array(self._get_state_value(next_states))
        states_value = np.array(self._get_state_value(states))

        TD_target : np.array = rewards + self.discount_factor * (1 - is_game_over) * next_states_value
        TD_error : np.array = TD_target - states_value

        self._update_policy_neural_network(states, actions, TD_error)
        self._update_state_value_neural_network(states, TD_target)

    def _update_policy_neural_network(self, states, actions, TD_error):
        self.policy_neural_network.compute_policy_log_probability_gradient(states, actions, TD_error)
        self.policy_neural_network.gradient_descent()

    def _update_state_value_neural_network(self, states, TD_target):
        self.state_value_neural_network.compute_state_value_gradients(states, TD_target)
        self.state_value_neural_network.gradient_descent()

    def get_ai_model(self):
        policy_neural_network_data = self.policy_neural_network.get_parameters()
        state_value_neural_network_data = self.state_value_neural_network.get_parameters()

        return {
            'policy_neural_network': policy_neural_network_data,
            'state_value_neural_network': state_value_neural_network_data,
        }