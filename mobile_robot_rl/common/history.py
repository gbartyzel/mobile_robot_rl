from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np

_FusionData = Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class BaseHistory:
    def __init__(self, time_length: int):
        self._time_length = time_length

    def __call__(self, *state: Any) -> Any:
        return state[0]

    def reset(self, *initial_state: Any):
        return initial_state[0]


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
                 image_size: Tuple[int, int]):
        super(VisionHistory, self).__init__(time_length)
        self._image_size = image_size
        self._state = np.zeros((self._time_length,) + image_size, np.uint8)

    def __call__(self, image_state: np.ndarray):
        self._state = np.roll(self._state, 1, 0)
        self._state[0, :] = self._perform_preprocessing(image_state)
        return self._state

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        image = self._perform_preprocessing(initial_state)
        self._state = np.stack((image,) * self._time_length)
        return self._state

    def _perform_preprocessing(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, self._image_size)
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        return grayscale_image.astype(np.uint8)


class FusionHistory(BaseHistory):
    def __init__(self,
                 time_length: int,
                 image_size: Tuple[int, int],
                 vector_dim: int,
                 ravel: bool = False,
                 vector_key: Optional[str] = None,
                 image_key: Optional[str] = None):
        super(FusionHistory, self).__init__(time_length)
        self._keys = [vector_key, image_key]
        self._image = VisionHistory(time_length, image_size)
        self._vector = VectorHistory(time_length, vector_dim, ravel)

    def __call__(self, state: _FusionData) -> _FusionData:
        if isinstance(state, dict):
            return {self._keys[0]: self._vector(state[self._keys[0]]),
                    self._keys[1]: self._image(state[self._keys[1]])}
        return self._vector(state[0]), self._image(state[1])

    def reset(self, state: _FusionData) -> _FusionData:
        if isinstance(state, dict):
            return {self._keys[0]: self._vector.reset(state[self._keys[0]]),
                    self._keys[1]: self._image.reset(state[self._keys[1]])}
        return self._vector.reset(state[0]), self._image.reset(state[1])
