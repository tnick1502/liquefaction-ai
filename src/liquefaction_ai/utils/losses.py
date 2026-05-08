from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target) ** 2, mask)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(torch.abs(pred - target), mask)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-6.0, max=3.0)
    inv_var = torch.exp(-logvar)
    return masked_mean(0.5 * (logvar + (target - mean) ** 2 * inv_var), mask)


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
