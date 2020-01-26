import abc
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import gym
import numpy as np
import torch
import torch.nn as nn
import tqdm

from mobile_robot_rl.common.logger import Logger
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
                 update_steps: int = 1,
                 use_soft_update: bool = False,
                 use_combined_experience_replay: bool = False,
                 logdir: str = './output',
                 seed: int = 1337):
        super(BaseOffPolicy, self).__init__()
        self.step = 0
        self._env = env
        self._state = None
        self._update_step = 0
        self._seed = seed
        self._set_seeds(seed)

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._action_dim = self._env.action_space.shape[0]

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

        state_dict_keys = None
        if isinstance(self._env.observation_space.spaces, dict):
            state_dict_keys = self._env.observation_space.spaces.keys()
        self._memory = ReplayMemory(
            capacity=memory_capacity,
            combined=use_combined_experience_replay,
            device=self._device,
            torch_backend=True,
            state_dict_keys=state_dict_keys)

        self._rollout = Rollout(length=n_step, discount_factor=discount_factor)
        self._logger = Logger(self, logdir)

    def _step(self, train: bool = False) -> Tuple[float, bool, Dict[str, bool]]:
        if train and self._memory.size < self._warm_up_steps:
            action = np.random.uniform(-1.0, 1.0, (2,))
        else:
            action = self._act(self._state, train)
        next_state, reward, done, info = self._env.step(action)
        if train:
            self._observe(self._state, action, reward, next_state, done)
        self._state = next_state
        return reward, done, info

    def train(self, max_steps: int, test_interval: int = 25000):
        self._state = self._env.reset()
        total_reward = []
        pb = tqdm.tqdm(total=max_steps + self._warm_up_steps)
        while self.step < max_steps:
            reward, done, info = self._step(True)
            total_reward.append(reward)
            pb.update(1)
            if done:
                self._state = self._env.reset()
                if self.step > 0:
                    self._logger.log_train(sum(total_reward),
                                           info['is_success'])
                total_reward = []
            if (self.step % test_interval) == (test_interval - 1):
                self._run_test()
        pb.close()
        self._env.close()
        self._logger.save_results()

    def eval(self, seed: int = None) -> Tuple[float, bool]:
        if seed is not None:
            self._set_seeds(seed)
        self._rollout.reset()
        self._state = self._env.reset()
        total_reward = 0.0
        while True:
            reward, done, info = self._step(False)
            total_reward += reward
            if done:
                break

        return total_reward, info['is_success']

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
            self.step += 1
            if self.step % self._update_frequency == 0:
                for _ in range(self._update_steps):
                    self._update_step += 1
                    self._update()
        if done:
            self._rollout.reset()

    def _run_test(self):
        results = [self.eval(i) for i in range(10)]
        rewards, success = zip(*results)
        self._logger.log_test(rewards, success)
        self._set_seeds(self._seed)
        self._rollout.reset()
        self._state = self._env.reset()
        self.step += 1

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

    def _convert_np_state(self, state):
        if isinstance(state, dict):
            return (
                torch.from_numpy(
                    np.array(state['scalars'])).float().unsqueeze(0).to(
                    self._device),
                torch.from_numpy(
                    np.array(state['image'])).float().unsqueeze(0).to(
                    self._device) / 255.0)
        return torch.from_numpy(np.array(state)).float().unsqueeze(0).to(
            self._device)

    def _convert_tensor_state(self, state):
        if isinstance(state, dict):
            return state['scalars'].float(), state['image'].float() / 255.0
        return state.float()

    @staticmethod
    def _hard_update(model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(model.state_dict())

    @staticmethod
    def _set_seeds(seed):
        torch.random.manual_seed(seed)
        np.random.seed(seed)

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
