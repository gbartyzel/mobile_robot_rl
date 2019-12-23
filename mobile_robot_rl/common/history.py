import numpy as np


class VectorHistory:
    def __init__(self, time_length: int, state_dim: int, ravel: bool = False):
        self._state = np.zeros((time_length, state_dim))

        self._time_length = time_length
        self._ravel = ravel

    def __call__(self, vector_state: np.ndarray) -> np.ndarray:
        self._state = np.roll(self._state, 1, 0)
        self._state[0, :] = vector_state
        if self._ravel:
            return self._state.ravel()
        return self._state

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        self._state = np.stack((initial_state,) * self._time_length)
        if self._ravel:
            return self._state.ravel()
        return self._state
