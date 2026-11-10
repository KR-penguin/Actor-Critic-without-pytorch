import config
from model import Agent

class Batch:
    def __init__(self, agent_instance : Agent):
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.is_game_over = []
        self.steps_in_batch = 0
        self.agent : Agent = agent_instance

    def is_full(self):

        return (self.steps_in_batch >= config.BATCH_SIZE)
    
    def add(self, action, state, reward, next_state, is_game_over):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.is_game_over.append(is_game_over)
        self.steps_in_batch += 1
    
    def clear(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.is_game_over = []
        self.steps_in_batch = 0