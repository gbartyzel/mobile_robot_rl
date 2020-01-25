from collections import deque
from typing import Optional
from typing import Tuple

import cv2
import gym
import gym.spaces as spaces
import gym.wrappers as wrappers
import numpy as np


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

    def observation(self, observation):
        if isinstance(observation, dict):
            image = observation[self._dict_key].copy()
        else:
            image = observation.copy()
        image = cv2.resize(image, self._image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, 0 if self._nchw else -1)
        if isinstance(observation, dict):
            observation[self._dict_key] = image.copy()
            return observation
        return image.copy()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, nchw):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.nchw = nchw

        if isinstance(self.observation_space, spaces.Dict):
            self.frames = dict()
            for key, value in self.observation_space.spaces.items():
                shp = value.shape
                if len(shp) == 1:
                    shp = (1, shp[0]) if nchw else (shp[0], 1)

                if nchw:
                    value.shape = ((shp[0] * k,) + shp[1:])
                else:
                    value.shape = (shp[:-1] + (shp[-1] * k,))
                self.observation_space.spaces[key] = value
                self.frames[key] = deque([], maxlen=k)
        else:
            self.frames = deque([], maxlen=k)
            shp = self.observation_space.shape
            if nchw:
                self.observation_space.shape = ((shp[0] * k,) + shp[1:])
            else:
                self.observation_space.shape = (shp[:-1] + (shp[-1] * k,))

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            if isinstance(ob, dict):
                for key, value in ob.items():
                    if len(value.shape) == 1:
                        value = np.expand_dims(value, 0 if self.nchw else -1)
                    self.frames[key].append(value)
            else:
                self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        if isinstance(self.frames, dict):
            obs = dict()
            for key, value in self.frames.items():
                assert len(value) == self.k
                obs[key] = LazyFrames(list(value))
            return obs
        else:
            assert len(self.frames) == self.k
            return LazyFrames(list(self.frames))


class LazyFrames:
    def __init__(self, frames, nchw: bool = True):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
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


def make_env(env):
    env = wrappers.RescaleAction(env, -1.0, 1.0)
    env = ConvertImage(env, (64, 64), True, dict_key='image')
    env = FrameStack(env, 4, True)
    return env


if __name__ == '__main__':
    frames = [np.random.rand(1, 64, 64),
              np.random.rand(1, 64, 64),
              np.random.rand(1, 64, 64),
              np.random.rand(1, 64, 64)]

    lframes = LazyFrames(frames)
    print(np.asarray(lframes).shape)
