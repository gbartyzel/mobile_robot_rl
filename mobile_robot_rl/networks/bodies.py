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


class FusionModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 input_channel: int,
                 hidden_dim: Tuple[int, ...]):
        super(FusionModel, self).__init__()
        self.output_dim = hidden_dim[-1]

        self._vision_body = nn.Sequential(
            nn.Conv2d(input_channel, 32, 5, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 3)
        )
        self._vision_output_dim = 32 * 3 * 3

        self._size = len(hidden_dim)
        self._body = nn.ModuleList()
        layers = (self._vision_output_dim + input_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self,
                x_vector: torch.Tensor,
                x_image: torch.Tensor) -> torch.Tensor:
        x_image = self._vision_body(x_image).view(-1, self._vision_output_dim)
        x = torch.cat((x_vector, x_image), dim=1)
        for layer in self._body:
            x = F.relu(layer(x))
        return x


class CriticFusionModel(nn.Module):
    def __init__(self,
                 action_dim: int,
                 hidden_dim: Tuple[int, ...],
                 fusion_model: FusionModel):
        super(CriticFusionModel, self).__init__()

        self._fusion_body = fusion_model

        self._size = len(hidden_dim)
        self._body = nn.ModuleList()
        layers = (self._fusion_body.output_dim + action_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(nn.Linear(layers[i], layers[i + 1]))

        self.output_dim = hidden_dim[-1] + self._fusion_body.output_dim

    def forward(self,
                x_action: torch.Tensor,
                x_vector: torch.Tensor,
                x_image: torch.Tensor) -> torch.Tensor:
        x_obs = self._fusion_body(x_vector, x_image)
        x = torch.cat((x_action, x_obs), dim=1)
        for layer in self._body:
            x = F.relu(layer(x))
        return x