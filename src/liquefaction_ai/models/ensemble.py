"""
Глубокий ансамбль (deep ensemble) одинаковых по архитектуре моделей.

Усредняет предсказания нескольких независимо обученных членов (разные сиды/инициализации).
Для траектории возвращает смешанную дисперсию ``E[var] + Var[mean]`` (эпистемическая +
алеаторная), что улучшает калибровку и proper-scoring (NLL/CRPS) и снижает разброс между
запусками. Реализует контракт ``forward_batch`` для совместимости с оценкой проекта.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

__all__ = ["EnsembleModel"]


class EnsembleModel(nn.Module):
    """Ансамбль моделей с единым интерфейсом ``forward_batch``."""

    def __init__(self, members: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def eval(self):
        for m in self.members:
            m.eval()
        return super().eval()

    def to(self, device):
        for m in self.members:
            m.to(device)
        return super().to(device)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outs = [m.forward_batch(batch) for m in self.members]
        result: Dict[str, torch.Tensor] = {}

        if "traj_mean" in outs[0]:
            means = torch.stack([o["traj_mean"] for o in outs], dim=0) # (K, B, T)
            mean = means.mean(dim=0)
            result["traj_mean"] = mean
            if "traj_logvar" in outs[0]:
                vars = torch.stack([torch.exp(o["traj_logvar"]) for o in outs], dim=0)
                mix_var = vars.mean(dim=0) + means.var(dim=0, unbiased=False) # алеаторная + эпистемическая
                result["traj_logvar"] = torch.log(mix_var.clamp_min(1e-6))

        if "risk_prob" in outs[0]:
            prob = torch.stack([o["risk_prob"] for o in outs], dim=0).mean(dim=0)
            prob = prob.clamp(1e-6, 1.0 - 1e-6)
            result["risk_prob"] = prob
            result["risk_logit"] = torch.log(prob / (1.0 - prob))

        for key in ["nliq_norm", "nliq", "crr", "g", "z"]:
            if key in outs[0]:
                result[key] = torch.stack([o[key] for o in outs], dim=0).mean(dim=0)
        return result
