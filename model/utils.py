import torch
import torch.nn as nn


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Update the parameters of `model_ema` as an exponential moving average (EMA)
    of `model`. Used to update the teacher encoder from the student."""
    for ema_param, param in zip(model_ema.parameters(), model.parameters()):
        ema_param.data = ema_param.data * m + param.data * (1.0 - m)


def deactivate_requires_grad(model: nn.Module):
    """Disable gradient computation for all parameters of a model (e.g. the EMA
    teacher branch, which is updated only via momentum)."""
    for param in model.parameters():
        param.requires_grad = False
