import copy
import numpy as np
import torch
from collections import namedtuple
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

Batch = namedtuple('Batch', 'state action reward next_state mask')
Dimension = namedtuple('Dimension', 'shape dtype')

_DATA_TYPE = Union[np.ndarray, torch.Tensor]
_DIM_TYPE = Union[int, Sequence[int]]
_DICT_DATE_TYPE = Union[Dict[str, _DATA_TYPE], _DATA_TYPE]
_DICT_DIM_TYPE = Union[Dimension, Dict[str, Dimension]]
_TRANSITION = Tuple[_DICT_DATE_TYPE, np.ndarray, _DICT_DATE_TYPE, float, bool]


class RingBuffer:
    TORCH_BACKEND = False

    def __init__(self,
                 capacity: int,
                 dimension: Dimension):
        self._size = 0
        self._dimension = (capacity,) + self._to_tuple(dimension.shape)
        self._dtype = dimension.dtype
        self._container = np.zeros(self._dimension, dimension.dtype)

    def add(self, value: Any):
        self._container[self._size, :] = value
        self._size = (self._size + 1) % self._dimension[0]

    def sample(self,
               idx: Union[Sequence[int], torch.Tensor],
               device: torch.device = torch.device('cpu')) -> _DATA_TYPE:
        batch = self._container[idx]
        if RingBuffer.TORCH_BACKEND:
            return torch.from_numpy(batch).float().to(device)
        return batch

    def reset(self):
        self._container = np.zeros(self._dimension, self._dtype)

    @property
    def size(self) -> int:
        return self._size

    @property
    def first(self) -> _DATA_TYPE:
        return self._container[0]

    @property
    def end(self) -> _DATA_TYPE:
        return self._container[-1]

    @property
    def data(self) -> _DATA_TYPE:
        return self._container

    @staticmethod
    def _to_tuple(value: Union[int, Sequence[int]]) -> Tuple[int, ...]:
        if isinstance(value, int):
            return value,
        return tuple(value)

    def __str__(self) -> str:
        return np.array2string(self._container)

    def __getitem__(self, item: int) -> np.ndarray:
        return self._container[item]


_STATE_BUFFER = Union[RingBuffer, Dict[str, RingBuffer]]


class ReplayMemory(object):
    def __init__(self,
                 capacity: int,
                 state_dim: _DICT_DIM_TYPE,
                 action_dim: Dimension,
                 reward_dim: Dimension = Dimension(1, np.float32),
                 terminal_dim: Dimension = Dimension(1, np.float32),
                 combined: bool = False,
                 torch_backend: bool = False):
        RingBuffer.TORCH_BACKEND = torch_backend
        self._capacity = capacity
        self._combined = combined
        self._size = 0

        if isinstance(state_dim, dict):
            self._state_buffer = {key: RingBuffer(capacity, value)
                                  for key, value in state_dim.items()}
        else:
            self._state_buffer = RingBuffer(capacity, state_dim)
        self._action_buffer = RingBuffer(capacity, action_dim)
        self._reward_buffer = RingBuffer(capacity, reward_dim)
        self._next_state_buffer = copy.deepcopy(self._state_buffer)
        self._mask_buffer = RingBuffer(capacity, terminal_dim)

    def push(self,
             state: _DICT_DATE_TYPE,
             action: _DATA_TYPE,
             reward: Union[float, torch.Tensor],
             next_state: _DICT_DATE_TYPE,
             terminal: Union[float, bool, torch.Tensor]):
        self._size = min(self._size + 1, self._capacity)

        self._state_buffer = self._push_state(self._state_buffer, state)
        self._action_buffer.add(action)
        self._reward_buffer.add(reward)
        self._next_state_buffer = self._push_state(self._next_state_buffer,
                                                   next_state)
        self._mask_buffer.add(1.0 - terminal)

    def sample(self,
               batch_size: int,
               device: torch.device = torch.device('cpu')) -> Batch:
        if self._combined:
            batch_size -= 1
        idxs = np.random.randint((self.size - 1), size=batch_size)
        if self._combined:
            idxs = np.append(idxs, np.array(self.size - 1, dtype=np.int32))

        batch = Batch(
            state=self._sample_state(self._state_buffer, idxs, device),
            action=self._action_buffer.sample(idxs, device),
            reward=self._reward_buffer.sample(idxs, device),
            next_state=self._sample_state(self._next_state_buffer, idxs,
                                          device),
            mask=self._mask_buffer.sample(idxs, device))

        return batch

    @property
    def size(self) -> int:
        return self._size

    def __getitem__(self, item: int) -> Tuple[np.ndarray, ...]:
        state = self._state_buffer[item]
        action = self._action_buffer[item]
        reward = self._reward_buffer[item]
        next_state = self._next_state_buffer[item]
        terminal = self._mask_buffer[item]
        return state, action, reward, next_state, terminal

    @staticmethod
    def _push_state(buffer: _STATE_BUFFER,
                    data: _DICT_DATE_TYPE) -> _STATE_BUFFER:
        if isinstance(buffer, dict):
            for key, value in data.items():
                buffer[key].add(value)
        else:
            buffer.add(data)
        return buffer

    @staticmethod
    def _sample_state(buffer: _STATE_BUFFER,
                      idxs: np.ndarray,
                      device: torch.device) -> _DICT_DATE_TYPE:
        if isinstance(buffer, dict):
            return {k: v.sample(idxs, device) for k, v in buffer.items()}
        return buffer.sample(idxs, device)


class Rollout:
    def __init__(self,
                 capacity: int,
                 state_dim: Union[Dimension, Dict[str, Dimension]],
                 action_dim: Dimension,
                 discount_factor: float):
        self._size = 0
        self._capacity = capacity
        self._discount_factor = discount_factor

        if isinstance(state_dim, dict):
            self._state_buffer = {key: self._create_buffer(value)
                                  for key, value in state_dim.items()}
        else:
            self._state_buffer = self._create_buffer(state_dim)
        self._action_buffer = self._create_buffer(action_dim)
        self._reward_buffer = self._create_buffer(Dimension(1, np.float32))

    @property
    def ready(self) -> int:
        return self._size == self._capacity

    def push(self,
             state: _DICT_DATE_TYPE,
             action: _DATA_TYPE,
             reward: Union[float, np.ndarray]):
        if isinstance(self._state_buffer, dict):
            for key in self._state_buffer.keys():
                self._state_buffer[key] = self._add(
                    self._state_buffer[key], state[key])
        else:
            self._state_buffer = self._add(self._state_buffer, state)
        self._action_buffer = self._add(self._action_buffer, action)
        self._reward_buffer = self._add(self._reward_buffer, reward)
        self._size = min(self._size + 1, self._capacity)

    def get_transition(self,
                       state: _DICT_DATE_TYPE,
                       action: _DATA_TYPE,
                       reward: Union[np.ndarray, float],
                       next_state: _DICT_DATE_TYPE,
                       done: Union[np.ndarray, bool]) -> _TRANSITION:
        self.push(state, action, reward)
        if self._size == self._capacity:
            cum_reward = self._compute_cumulative_reward()
            if isinstance(self._state_buffer, dict):
                state = {key: val[0] for key, val in self._state_buffer.items()}
            else:
                state = self._state_buffer[0]
            action = self._action_buffer[0]
            return state, action, cum_reward, next_state, done
        return None

    def reset(self):
        if isinstance(self._state_buffer, dict):
            self._state_buffer = {key: np.zeros(self._state_buffer[key].shape)
                                  for key in self._state_buffer.keys()}
        else:
            self._state_buffer = np.zeros(self._state_buffer.shape)
        self._action_buffer = np.zeros(self._action_buffer.shape)
        self._reward_buffer = np.zeros(self._reward_buffer.shape)
        self._size = 0

    def _compute_cumulative_reward(self) -> float:
        cum_reward = 0
        for t in range(self._capacity):
            cum_reward += self._discount_factor ** t * self._reward_buffer[t]
        return cum_reward

    @staticmethod
    def _add(buffer: np.ndarray, value: Any) -> np.ndarray:
        buffer = np.roll(buffer, -1, 0)
        buffer[-1] = value
        return buffer

    def _create_buffer(self, dim: Dimension) -> _DICT_DATE_TYPE:
        if isinstance(dim.shape, tuple):
            return np.zeros((self._capacity,) + dim.shape, dim.dtype)
        return np.zeros((self._capacity, dim.shape), dim.dtype)


if __name__ == '__main__':
    state_dim = Dimension(shape=(4, 128, 128), dtype=np.uint8)
    action_dim = Dimension(shape=2, dtype=np.float32)

    memory = ReplayMemory(100, state_dim, action_dim)
    rollout = Rollout(5, state_dim, action_dim, 0.99)
