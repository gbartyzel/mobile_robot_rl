from typing import Union

import numpy as np
import torch


class GaussianNoise:
    TORCH_BACKEND = True

    def __init__(self,
                 dim: int,
                 mean: float = 0.0,
                 sigma: float = 1.0,
                 sigma_min: float = 0.0,
                 n_step_annealing: float = 1e6):
        self._dim = dim
        self._mean = mean
        self._sigma = sigma

        self._sigma_min = sigma_min
        if n_step_annealing == 0:
            self._sigma_decay_factor = 0
        else:
            self._sigma_decay_factor = (sigma - sigma_min) / n_step_annealing

    def reset(self):
        return

    def _reduce_sigma(self):
        self._sigma -= self._sigma_decay_factor
        self._sigma = max(self._sigma, self._sigma_min)

    def __call__(self) -> Union[np.ndarray, torch.Tensor]:
        self._reduce_sigma()
        if GaussianNoise.TORCH_BACKEND:
            return torch.normal(mean=self._mean,
                                std=torch.ones(self._dim) * self._sigma)
        return np.random.normal(loc=self._mean, scale=self._sigma,
                                size=self._dim)


class OUNoise(GaussianNoise):
    """
    Ornstein–Uhlenbeck process implementation
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self,
                 dim: int,
                 mean: float = 0.0,
                 theta: float = 0.15,
                 sigma: float = 0.2,
                 sigma_min: float = 0.0,
                 n_step_annealing: float = 1e6,
                 dt: float = 1e-2):
        """
        :param dim: int, dimension of ou process
        :param mean: float, asymptotic mean
        :param theta: float, define how 'strongly' systems react to
        perturbations
        :param sigma: float, the variation of the noise
        :param sigma_min: float, minimal value of the variation
        :param n_step_annealing: float, decremental steps for sigma
        :param dt: float,
        """
        super(OUNoise, self).__init__(dim, mean, sigma, sigma_min,
                                      n_step_annealing)
        self._dt = dt
        self._theta = theta

        self._state = torch.ones(self._dim) * self._mean
        self.reset()

    def reset(self):
        """
        Reset state of the noise
        """
        if GaussianNoise.TORCH_BACKEND:
            self._state = torch.ones(self._dim) * self._mean
        else:
            self._state = np.ones(self._dim) * self._mean

    def __call__(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Calculate noise value on the step t
        :return: np.ndarray, noise
        """
        self._reduce_sigma()
        if GaussianNoise.TORCH_BACKEND:
            noise = torch.normal(mean=torch.zeros(self._dim))
        else:
            noise = np.random.normal(loc=np.zeros(self._dim))
        x = (self._state + self._theta * (self._mean - self._state) * self._dt
             + np.sqrt(self._dt) * self._sigma * noise)
        self._state = x
        return x
