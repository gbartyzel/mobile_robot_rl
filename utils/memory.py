import random
import numpy as np
from collections import deque


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque()

    def add(self, state, action, reward, next_state, done):
        experience = state + action + [reward] + next_state + [done]
        if self.get_size() >= self.capacity:
            self.buffer.popleft()
        self.buffer.append(experience)

    def sample(self, batch_size):
        return np.array(random.sample(self.buffer, batch_size))
    
    @property
    def size(self):
        return len(self.buffer)