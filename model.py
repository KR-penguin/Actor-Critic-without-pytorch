class Agent:
    def __init__(self):
        self.discount_factor = 0.98
        self.last_state = None
        self.last_action = None
        self.policy_neural_network = None
        self.approximated_state_value_function_neural_network = None

    def _get_state_value(state):
        approximated_state_value_function_neural_network.get_approximated_state_value_function_value(state)

    def get_action_from_policy(self, state):
        actions_probabilities = self.policy_neural_network.get_actions_probabilities(state)
        '''
        example output
        {
            NO_OPERATION : 0.7,
            EXCUTION_LEFT_ENGINE : 0.2,
            EXCUTION_MAIN_ENGINE :0.05,
            EXCUTION_RIGHT_ENGINE : 0.05
        }
        '''
        action = max(action_probabilities, key=action_probabilities.get)
        last_action = action
        return action

    def update(self, action, state, reward):
        TD_target = reward + discount_factor * _get_state_value(state)
        last_state_value = _get_state_value(last_state)
        TD_error = TD_target - last_state_value
        _update_policy(TD_error)
        _update_approximated_state_value_function(TD_target, last_state_value)
        last_state = state

    def _update_policy(self, TD_error):
        gradient = policy_neural_network.get_policy_log_probability_gradient(self.last_action, self.last_state)
        objective_function_gradient = gradient * TD_error
        self.policy_neural_network.gradient_ascent(objective_function_gradient)

    def _update_approximated_state_value_function(self, TD_target, last_state_value):
        self.approximated_state_value_function_neural_network.update_approximated_state_value_function(TD_target, last_state_value)