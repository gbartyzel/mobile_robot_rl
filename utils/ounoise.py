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
        self.state += (
            self.theta * (self.mu - self.state)
            + self.sigma * np.random.randn(len(self.state)))

        return self.state

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
