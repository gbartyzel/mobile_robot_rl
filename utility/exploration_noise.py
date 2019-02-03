import numpy as np
import numpy.random as nr


class GaussianNoise(object):
    def __init__(self, dim: int, mean: float = 0.0, sigma: float = 1.0):
        self._dim = dim
        self._mean = mean
        self._sigma = sigma

    def __call__(self):
        return nr.normal(self._mean, self._sigma, (self._dim,))


class AdaptiveGaussianNoise(GaussianNoise):
    def __init__(self, dim: int, mean: float = 0.0, sigma: float = 1.0, sigma_min: float = 0.0,
                 n_step_annealing: float = 1e6):
        super(AdaptiveGaussianNoise, self).__init__(dim, mean, sigma)

        self._sigma_min = sigma_min
        self._sigma_decay_factor = (sigma - sigma_min) / n_step_annealing

    def _reduce_sigma(self):
        self._sigma -= self._sigma_decay_factor
        self._sigma = max(self._sigma, self._sigma_min)

    def __call__(self):
        self._reduce_sigma()
        return nr.normal(self._mean, self._sigma, (self._dim,))


class OUNoise(AdaptiveGaussianNoise):
    """
    Ornstein–Uhlenbeck process implementation
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, dim: int, mean: float = 0.0, theta: float = 0.15, sigma: float = 0.2,
                 sigma_min: float = 0.0, n_step_annealing: float = 1e6, dt: float = 1e-2):
        super(OUNoise, self).__init__(dim, mean, sigma, sigma_min, n_step_annealing)
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
        self._dt = dt
        self._theta = theta

        self._state = np.ones(self._dim) * self._mean
        self.reset()

    def reset(self):
        """
        Reset state of the noise
        """
        self._state = np.ones(self._dim) * self._mean

    def __call__(self) -> np.ndarray:
        """
        Calculate noise value on the step t
        :return: np.ndarray, noise
        """
        self._reduce_sigma()

        x = (self._state + self._theta * (self._mean - self._state) * self._dt
             + np.sqrt(self._dt) * self._sigma * nr.normal(size=self._state.shape))
        self._state = x
        return x
