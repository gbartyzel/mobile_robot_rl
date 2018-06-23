import numpy as np
import numpy.random as nr


class OUNoise:
    """
    Ornstein–Uhlenbeck process implementation
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """
    def __init__(self, dim, mu=0.0, theta=0.15, sigma=0.2,
                 sigma_min=0.0, n_step_annealing=1e6, dt=1e-2):
        """
        :param dim: int, dimension of ou process
        :param mu: float, asymptotic mean
        :param theta: float, define how 'strongly' systems react to
        perturbations
        :param sigma: float, the variation of the noise
        :param sigma_min: float, minimal value of the variation
        :param n_step_annealing: float, decremental steps for sigma
        :param dt: float,
        """
        self._dim = dim
        self._dt = dt
        self._mu = mu
        self._n_step_annealing = sigma / n_step_annealing

        self._theta = theta
        self._sigma = sigma
        self._sigma_min = sigma_min

        self._state = np.ones(self._dim) * self._mu
        self.reset()

    def reset(self):
        """
        Reset state of the noise
        """
        self._state = np.ones(self._dim) * self._mu

    def noise(self):
        """
        Calculate noise value on the step t
        :return: np.ndarray, noise
        """
        self._sigma -= self._n_step_annealing
        self._sigma = max(self._sigma_min, self._sigma)

        x = (self._state + self._theta * (self._mu - self._state) * self._dt
             + np.sqrt(self._dt) * self._sigma
             * nr.normal(size=self._state.shape))
        self._state = x
        return x
