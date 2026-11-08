import config
from model import Agent

class Batch:
    def __init__(self, agent_instance : Agent):
        self.actions = []
        self.states = []
        self.rewards = []
        self.steps_in_batch = 0
        self.batch_size = config.BATCH_SIZE
        self.agent : Agent = agent_instance

    def is_full(self):
        return (self.steps_in_batch > config.BATCH_SIZE)
    
    def add(self, action, state, reward):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.steps_in_batch += 1
    
    def clear(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.steps_in_batch = 0