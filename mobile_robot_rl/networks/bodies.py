from typing import Callable
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm


def fan_init(x: nn.Parameter):
    size = x.data.size()[1]
    val = 1 / np.sqrt(size)
    return -val, val


def orthogonal_init(x: nn.Module):
    classname = x.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
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

    def forward(self, *x: Sequence[torch.Tensor]) -> torch.Tensor:
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
                 input_channel: int,
                 hidden_dim: Tuple[int, ...]):
        super(FusionModel, self).__init__()
        self.output_dim = hidden_dim[-1]

        self._vision_body = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )
        self._vision_output_dim = 512

        self._vector_body = nn.Sequential(
            nn.Conv1d(input_channel, 16, 2, 1, 1, dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 16, 2, 1, 1, dilation=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 16, 2, 1, 1, dilation=4),
            nn.LeakyReLU(inplace=True),
        )
        self._vector_output_dim = 208

        self._size = len(hidden_dim)
        self._body = nn.ModuleList()
        layers = (self._vision_output_dim +
                  self._vector_output_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(nn.Linear(layers[i], layers[i + 1]))

        self.reset_parameters()

    def forward(self,
                x_vector: torch.Tensor,
                x_image: torch.Tensor) -> torch.Tensor:
        x_image = self._vision_body(x_image).view(-1, self._vision_output_dim)
        x_vector = self._vector_body(x_vector).view(-1, self._vector_output_dim)
        x = torch.cat((x_vector, x_image), dim=1)
        for i in range(len(self._body)):
            x = F.leaky_relu(self._body[i](x))
        return x

    def reset_parameters(self):
        for layer in self._body:
            orthogonal_init(layer)
        self._vision_body.apply(orthogonal_init)
        self._vector_body.apply(orthogonal_init)


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

        self.output_dim = hidden_dim[-1]

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self._body:
            orthogonal_init(layer)

    def forward(self,
                x_action: torch.Tensor,
                x_vector: torch.Tensor,
                x_image: torch.Tensor) -> torch.Tensor:
        x_obs = self._fusion_body(x_vector, x_image)
        x = torch.cat((x_obs, x_action), dim=1)
        for i in range(len(self._body)):
            x = F.leaky_relu(self._body[i](x))
        return x

if __name__ == '__main__':
    model = FusionModel(4, (512, ))
    model(torch.rand(1, 4, 14), torch.rand(1, 4, 64, 64))