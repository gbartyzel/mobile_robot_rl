import numpy as np


class OUNoise(object):

    def __init__(self, config):
        self.action_dim = config.action_dim
        self.mu = config.mu
        self.sigma = config.sigma
        self.theta = config.theta

        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def noise(self):
        x = self.state
        dx = (self.theta * (self.mu - x)
              + self.sigma * np.random.randn(len(x)))
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
