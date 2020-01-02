from typing import Any
from typing import Dict
from typing import Tuple

import cv2
import numpy as np


class BaseHistory:
    def __init__(self, time_length: int):
        self._time_length = time_length

    def __call__(self, state: Any) -> Any:
        return NotImplementedError

    def reset(self, initial_state: Any):
        return NotImplementedError


class VectorHistory(BaseHistory):
    def __init__(self, time_length: int, state_dim: int, ravel: bool = False):
        super(VectorHistory, self).__init__(time_length)
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


class VisionHistory(BaseHistory):
    def __init__(self,
                 time_length: int,
                 resize: Tuple[int, int]):
        super(VisionHistory, self).__init__(time_length)
        self._resize = resize
        self._state = np.zeros((self._time_length,) + resize, np.uint8)

    def __call__(self, image_state: np.ndarray):
        self._state = np.roll(self._state, 1, 0)
        self._state[0, :] = self._perform_preprocessing(image_state)
        return self._state

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        image = self._perform_preprocessing(initial_state)
        self._state = np.stack((image,) * self._time_length)
        return self._state

    def _perform_preprocessing(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, self._resize)
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        return grayscale_image.astype(np.uint8)


class FusionHistory:
    def __init__(self,
                 time_length,
                 ravel,
                 image_size,
                 vector_dim,
                 keys: Tuple[str, str]):
        self._image = VisionHistory(time_length, image_size)
        self._scalars = VectorHistory(time_length, vector_dim, ravel)

    def __call__(self, state) -> Dict[str, np.ndarray]:
        pass
