from typing import Callable
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fan_init(x: nn.Parameter):
    size = x.data.size()[1]
    val = 1 / np.sqrt(size)
    return -val, val


def orthogonal_init(x: nn.Module):
    classname = x.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.orthogonal_(x.weight.data, gain=np.sqrt(2))
        nn.init.constant_(x.bias.data, 0.0)


class BaseMLPNetwork(nn.Module):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 output_dim: int,
                 activation_fn: Union[Callable, nn.Module] = F.relu):
        super(BaseMLPNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._activation_fn = activation_fn

    def forward(self, *input):
        return NotImplementedError


class MLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: nn.Module = F.relu):
        super(MLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                         activation_fn)

        self._size = len(hidden_dim)
        self._body = nn.ModuleList()
        layers = (input_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(nn.Linear(layers[i], layers[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self._body:
            orthogonal_init(layer)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        for layer in self._body:
            x = self._activation_fn(layer(x))
        return x


class BNMLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: nn.Module = F.relu):
        super(BNMLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                           activation_fn)

        self._size = len(hidden_dim)
        self._body = nn.ModuleList()
        layers = (input_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(nn.Linear(layers[i], layers[i + 1]))
            self._body.append(nn.BatchNorm1d(layers[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self._body:
            orthogonal_init(layer)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        for i in range(len(self._body) // 2):
            x = self._body[2 * i + 1](self._body[2 * i](x))
            x = self._activation_fn(x)
        return x


class TimeSeriesNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 activation_fn: Callable = F.relu):
        super(TimeSeriesNetwork, self).__init__()
        self._activation_fn = activation_fn

        self._body = nn.Sequential(
            nn.Conv1d(input_dim, 16, 3, padding=1),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, 2),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, 2),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
        )
        self.output_dim = 112

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._body(x)
        x = x.view(-1, self.output_dim)
        return x


class TimeSeriesCriticNetwork(nn.Module):
    def __init__(self,
                 input_dim: Tuple[int, int],
                 activation_fn: Callable = F.relu):
        super(TimeSeriesCriticNetwork, self).__init__()
        self._activation_fn = activation_fn

        self._body = nn.Sequential(
            nn.Conv1d(input_dim[0], 16, 3, padding=1),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, 2),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, 2),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
        )
        self._output_dim = 112

        self.output_dim = 128

        self._fc_1 = nn.Linear(input_dim[1] + self._output_dim, 128)
        self._fc_2 = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x[1]
        x = self._body(x[0])
        x = x.view(-1, self._output_dim)
        x = F.relu(self._fc_1(torch.cat((x, u), 1)))
        x = F.relu(self._fc_2(x))
        return x
