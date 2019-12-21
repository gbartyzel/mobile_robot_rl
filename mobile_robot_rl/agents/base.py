import abc
from typing import Any
from typing import Tuple
from typing import Union

import gym
import numpy as np
import torch
import torch.nn as nn
import tqdm
from gym.spaces.box import Box
from torch.utils.tensorboard import SummaryWriter

from mobile_robot_rl.common.memory import ReplayMemory
from mobile_robot_rl.common.memory import Rollout


class BaseOffPolicy(abc.ABC):
    def __init__(self,
                 env: gym.Env,
                 discount_factor: float = 0.99,
                 n_step: int = 1,
                 memory_capacity: int = int(1e5),
                 batch_size: int = 64,
                 warm_up_steps: int = 64,
                 reward_scaling: float = 1.0,
                 polyak_factor: float = 0.001,
                 update_frequency: int = 1,
                 target_update_frequency: int = 1000,
                 update_steps: int = 4,
                 use_soft_update: bool = False,
                 use_combined_experience_replay: bool = False,
                 logdir: str = './output',
                 seed: int = 1337):
        super(BaseOffPolicy, self).__init__()
        self._env = env
        self._state = None
        self._step = 0
        self._update_step = 0
        self._set_seeds(seed)

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._state_dim = int(np.prod(self._env.observation_space.shape))
        if isinstance(self._env.action_space, Box):
            self._action_dim = self._env.action_space.shape[0]
            memory_action = self._action_dim
        else:
            self._action_dim = self._env.action_space.n
            memory_action = 1

        self._discount = discount_factor ** n_step
        self._n_step = n_step
        self._reward_scaling = reward_scaling

        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps

        self._use_soft_update = use_soft_update
        self._polyak = polyak_factor
        self._update_frequency = update_frequency
        self._target_update_frequency = target_update_frequency

        self._update_steps = update_steps

        self._memory = ReplayMemory(
            capacity=memory_capacity,
            state_dim=self._state_dim,
            action_dim=memory_action,
            combined=use_combined_experience_replay,
            torch_backend=True)

        self._rollout = Rollout(
            capacity=n_step,
            state_dim=self._state_dim,
            action_dim=memory_action,
            discount_factor=discount_factor)

        self._writer = SummaryWriter(logdir)

    def step(self, train: bool):
        if train and self._memory.size < self._warm_up_steps:
            action = np.random.uniform(-1.0, 1.0, (2, ))
        else:
            action = self._act(self._state, train)
        next_state, reward, done, _ = self._env.step(action)
        self._observe(self._state, action, reward, next_state, done)
        self._state = next_state
        return reward, done

    def train(self, max_steps: int):
        self._state = self._env.reset()
        total_reward = []
        pb = tqdm.tqdm(total=max_steps)
        while self._step < max_steps:
            reward, done = self.step(True)
            total_reward.append(reward)
            pb.update(1)
            if done:
                self._state = self._env.reset()
                if self._step > 0:
                    self._log(sum(total_reward))
                total_reward = []
        pb.close()

    def eval(self, render: bool = False):
        self._state = self._env.reset()
        while True:
            if render:
                self._env.render()
            _, done = self.step(False)
            if done:
                break

    def _observe(self,
                 state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 reward: Union[float, torch.Tensor],
                 next_state: Union[np.ndarray, torch.Tensor],
                 done: Any):
        transition = self._rollout.get_transition(state, action, reward,
                                                  next_state, done)
        if transition is None:
            return
        self._memory.push(*transition)
        if self._memory.size >= self._warm_up_steps:
            self._step += 1
            if self._step % self._update_frequency == 0:
                for _ in range(self._update_steps):
                    self._update_step += 1
                    self._update()
        if done:
            self._rollout.reset()

    def _update_target(self, model: nn.Module, target_model: nn.Module):
        if self._use_soft_update:
            self._soft_update(model.parameters(), target_model.parameters())
        else:
            if self._update_step % self._target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict())

    def _soft_update(self, params: nn.parameter, target_params: nn.parameter):
        for param, t_param in zip(params, target_params):
            t_param.data.copy_(
                t_param.data * (1.0 - self._polyak) + param.data * self._polyak)

    def _td_target(self,
                   reward: torch.Tensor,
                   mask: torch.Tensor,
                   next_value: torch.Tensor) -> torch.Tensor:
        return reward + mask * self._discount * next_value

    @staticmethod
    def _hard_update(model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(model.state_dict())

    @staticmethod
    def _set_seeds(seed):
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    def _log(self, reward):
        self._writer.add_scalar('reward', reward, self._step)
        for name, param in self.parameters:
            self._writer.add_histogram(
                'main/{}'.format(name), param, self._step)
        for name, param in self.target_parameters:
            self._writer.add_histogram(
                'target/{}'.format(name), param, self._step)

    @property
    @abc.abstractmethod
    def parameters(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def target_parameters(self):
        return NotImplementedError

    @abc.abstractmethod
    def _act(self, *args):
        return NotImplementedError

    @abc.abstractmethod
    def _update(self):
        return NotImplementedError

    @abc.abstractmethod
    def load(self, path: str):
        return NotImplementedError

    @abc.abstractmethod
    def save(self):
        return NotImplementedError
