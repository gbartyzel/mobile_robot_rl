from copy import deepcopy
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobile_robot_rl.networks.heads import DeterministicPolicyHead
from mobile_robot_rl.networks.heads import GaussianPolicyHead
from mobile_robot_rl.networks.heads import ValueHead


class Critic(nn.Module):
    def __init__(self, phi: nn.Module):
        super(Critic, self).__init__()
        self._phi = deepcopy(phi)
        self._value = ValueHead(self._phi.output_dim)

    def forward(self, *x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._value(self._phi(x))


class DistributionalCritic(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 distribution_type: str,
                 support_dim: int):
        super(DistributionalCritic, self).__init__()
        assert distribution_type in ('categorical', 'quantile')

        self._phi = deepcopy(phi)
        self._dist = ValueHead(self._phi.output_dim, support_dim)
        self._support_dim = support_dim
        self._distribution_type = distribution_type

    def forward(self, *x):
        probs = self._dist(x).view(-1, 1, self._support_dim)
        if self._distribution_type == 'categorical':
            return F.softmax(probs, dim=-1)
        return probs


class DoubleCritic(nn.Module):
    def __init__(self, phi: Union[Tuple[nn.Module, nn.Module], nn.Module]):
        super(DoubleCritic, self).__init__()
        if isinstance(phi, tuple):
            self._critic_1 = Critic(phi[0])
            self._critic_2 = Critic(phi[1])
        else:
            self._critic_1 = Critic(phi)
            self._critic_2 = Critic(phi)

    def q1_parameters(self):
        return self._critic_1.parameters()

    def q2_parameters(self):
        return self._critic_2.parameters()

    def forward(self,
                *x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self._critic_1(*x), self._critic_2(*x)


class DeterministicActor(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 fan_init: bool = False,
                 activation_fn: Callable = torch.tanh):
        super(DeterministicActor, self).__init__()
        self._phi = deepcopy(phi)
        self._head = DeterministicPolicyHead(
            input_dim=self._phi.output_dim,
            output_dim=output_dim,
            fan_init=fan_init,
            activation_fn=activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self._phi(x))


class GaussianActor(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 std_limits: Tuple[float, float] = (-20.0, 2.0),
                 independent_std: bool = False,
                 squash: bool = True,
                 reparameterize: bool = True,
                 fan_init: bool = True):
        super(GaussianActor, self).__init__()
        self._phi = phi
        self._head = GaussianPolicyHead(
            input_dim=self._phi.output_dim,
            output_dim=output_dim,
            std_limits=std_limits,
            independent_std=independent_std,
            squash=squash,
            reparameterize=reparameterize,
            fan_init=fan_init)

    def forward(self,
                x: torch.Tensor,
                raw_action: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, ...]:
        return self._head.sample(self._phi(x), raw_action, deterministic)
