import numpy as np
import numpy.random as nr


class OUNoise:
    def __init__(
            self, action_dim, mu=0, theta=0.15, sigma=0.3, sigma_min=0.0,
            n_step_annealing=1e6, dt=1e-2):
        self.action_dimension = action_dim
        self.dt = dt
        self.mu = mu
        self.n_step_annealing = sigma / n_step_annealing

        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min

        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        self.sigma -= self.n_step_annealing
        self.sigma = max(self.sigma_min, self.sigma)

        x = self.state
        dx = (self.theta * (self.mu - x) * self.dt
              + np.sqrt(self.dt) * self.sigma * nr.randn(len(x)))
        self.state = x + dx
        return self.state

