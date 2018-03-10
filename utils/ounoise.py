import numpy as np
import numpy.random as nr


class OUNoise:
    def __init__(self,
                 action_dim,
                 mu=0,
                 theta=0.15,
                 sigma=0.3,
                 sigma_min=0.0,
                 n_step_annealing=1e6,
                 dt=1e-2):
        self._action_dim = action_dim
        self._dt = dt
        self._mu = mu
        self._n_step_annealing = sigma / n_step_annealing

        self._theta = theta
        self._sigma = sigma
        self._sigma_min = sigma_min

        self._state = np.ones(self._action_dim) * self._mu
        self.reset()

    def reset(self):
        self._state = np.ones(self._action_dim) * self._mu

    def noise(self):
        self._sigma -= self._n_step_annealing
        self._sigma = max(self._sigma_min, self._sigma)

        self._state += (
            self._theta * (self._mu - self._state) * self._dt +
            np.sqrt(self._dt) * self._sigma * nr.randn(len(self._state)))
        return self._state
