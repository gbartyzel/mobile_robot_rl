from collections import deque
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import gym
import gym.spaces as spaces
import gym.wrappers as wrappers
import numpy as np


class LazyFrames:
    def __init__(self, frames: List[Any], nchw: bool = True):
        self._frames = frames
        self._out = None
        self._nchw = nchw

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames,
                                       axis=0 if self._nchw else -1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


_OBSERVATION_TYPE = Union[np.ndarray, Dict[str, np.ndarray]]
_FRAME_STACK_TYPE = Union[LazyFrames, Dict[str, LazyFrames]]
_TRANSITION = Tuple[_FRAME_STACK_TYPE, float, bool, Dict[str, Any]]


class ConvertImage(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 image_size: Tuple[int, int],
                 nchw: bool = True,
                 dict_key: Optional[str] = None):
        super().__init__(env)
        self._image_size = image_size
        self._dict_key = dict_key
        self._nchw = nchw

        image_shape = (1,) + image_size if nchw else image_size + (1,)
        image_space = spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8)

        if dict_key is not None:
            self.observation_space.spaces[dict_key] = image_space
        else:
            self.observation = image_space

    def observation(self, observation: _OBSERVATION_TYPE) -> _OBSERVATION_TYPE:
        if isinstance(observation, dict):
            image = observation[self._dict_key].copy()
        else:
            image = observation.copy()
        image = cv2.resize(image, self._image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if isinstance(observation, dict):
            observation[self._dict_key] = np.expand_dims(image, self._nchw - 1)
            return observation
        return np.expand_dims(image, self._nchw - 1)


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 1, nchw: bool = True):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.nchw = nchw

        def generate_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
            if nchw:
                return (shape[0] * k,) + shape[1:]
            return shape[:-1] + (shape[-1] * k,)

        if isinstance(self.observation_space, spaces.Dict):
            self.frames = dict()
            for key, value in self.observation_space.spaces.items():
                shp = value.shape
                if len(shp) == 1:
                    shp = (1, shp[0]) if nchw else (shp[0], 1)
                value.shape = generate_shape(shp)
                self.observation_space.spaces[key] = value
                self.frames[key] = deque([], maxlen=k)
        else:
            self.frames = deque([], maxlen=k)
            shp = self.observation_space.shape
            self.observation_space.shape = generate_shape(shp)

    def reset(self) -> _FRAME_STACK_TYPE:
        ob = self.env.reset()
        for _ in range(self.k):
            self._append_observation(ob)
        return self._get_ob()

    def step(self, action: Union[np.ndarray, int, float]) -> _TRANSITION:
        ob, reward, done, info = self.env.step(action)
        self._append_observation(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self) -> _FRAME_STACK_TYPE:
        if isinstance(self.frames, dict):
            obs = dict()
            for key, value in self.frames.items():
                assert len(value) == self.k
                obs[key] = LazyFrames(list(value))
            return obs
        else:
            assert len(self.frames) == self.k
            return LazyFrames(list(self.frames))

    def _append_observation(self, observation: _OBSERVATION_TYPE):
        if isinstance(observation, dict):
            for key, value in observation.items():
                self.frames[key].append(np.expand_dims(value, self.nchw - 1)
                                        if len(value.shape) == 1 else value)
        else:
            self.frames.append(observation)


def make_env(env: gym.Env,
             length: int = 1,
             action_limits: Tuple[int, int] = (-1.0, 1.0),
             image_size: Tuple[int, int] = (64, 64),
             nchw_format: bool = True):
    env = wrappers.RescaleAction(env, *action_limits)
    env = ConvertImage(env, image_size, nchw_format, dict_key='image')
    env = FrameStack(env, length, nchw_format)
    return env
