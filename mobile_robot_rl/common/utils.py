from enum import Enum

import torch


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def mse_loss(prediction: torch.Tensor,
             target: torch.Tensor,
             reduction: Reduction = Reduction.MEAN) -> torch.Tensor:
    loss = 0.5 * (prediction - target).pow(2)

    if reduction == Reduction.MEAN:
        return torch.mean(loss)
    elif reduction == Reduction.SUM:
        return torch.sum(loss)
    return loss


def huber_loss(prediction: torch.Tensor,
               target: torch.Tensor,
               delta: float = 1.0,
               reduction: Reduction = Reduction.MEAN) -> torch.Tensor:
    error = target - prediction
    loss = torch.where(torch.abs(error) < delta,
                       0.5 * error.pow(2),
                       delta * (error.abs() - 0.5 * delta))
    if reduction == Reduction.MEAN:
        return torch.mean(loss)
    elif reduction == Reduction.SUM:
        return torch.sum(loss)
    return loss


def quantile_hubber_loss(prediction: torch.Tensor,
                         target: torch.Tensor,
                         cumulative_density: torch.Tensor,
                         delta: float = 1.0,
                         reduction: Reduction = Reduction.MEAN) -> torch.Tensor:
    transpose_target = target.t().unsqueeze(-1)
    diff = transpose_target - prediction
    loss = huber_loss(prediction, transpose_target, delta, Reduction.NONE)
    loss *= torch.abs(cumulative_density - (diff < 0.0).float())
    loss = loss.mean(0).sum(1)

    if reduction == Reduction.MEAN:
        return torch.mean(loss)
    elif reduction == Reduction.SUM:
        return torch.sum(loss)
    return loss
