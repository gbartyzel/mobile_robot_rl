import os
import datetime
from typing import List

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self,
                 agent,
                 log_dir: str = './output'):
        self._agent = agent
        self._log_dir = self._create_logdir(log_dir)
        self._train_writer = SummaryWriter(os.path.join(self._log_dir, 'train'))
        self._test_writer = SummaryWriter(os.path.join(self._log_dir, 'test'))

        self._sucess_rate_container = list()
        self._reward_container = list()
        self._step_container = list()

    def log_train(self, reward: float, success: bool):
        self._train_writer.add_scalar('reward/mean', reward, self._agent.step)
        self._train_writer.add_scalar('success_rate', success, self._agent.step)

        for name, param in self._agent.parameters:
            self._train_writer.add_histogram(
                'main/{}'.format(name), param, self._agent.step)
        for name, param in self._agent.target_parameters:
            self._train_writer.add_histogram(
                'target/{}'.format(name), param, self._agent.step)

    def log_test(self, rewards: List[float], success_rates: List[bool]):
        self._sucess_rate_container.append(success_rates)
        self._reward_container.append(rewards)
        self._step_container.append(self._agent.step)

        self._test_writer.add_scalar(
            'reward/mean', np.mean(rewards), self._agent.step)
        self._test_writer.add_scalar('reward/min', np.min(rewards),
                                     self._agent.step)
        self._test_writer.add_scalar('reward/max', np.max(rewards),
                                     self._agent.step)
        self._test_writer.add_scalar('reward/std', np.std(rewards),
                                     self._agent.step)
        self._test_writer.add_scalar('success_rate', np.mean(success_rates),
                                     self._agent.step)
        self._agent.save(self._log_dir)

    def save_results(self):
        data = zip(self._step_container, self._reward_container,
                   self._sucess_rate_container)
        df = pd.DataFrame(data, columns=['steps', 'rewards', 'success_rate'])
        df.to_csv(os.path.join(self._log_dir, 'results.csv'))

    @staticmethod
    def _create_logdir(log_dir: str) -> str:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        now = datetime.datetime.now().strftime('%d_%m_%y_%H_%M_%S')
        return os.path.join(log_dir, now)
