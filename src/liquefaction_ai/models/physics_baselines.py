"""
Физически-информированный baseline (PINN) для кривой порового давления PPR(N).

:class:`PINNBaseline` — координатная сеть, предсказывающая PPR(n) по статическим признакам и
координатам нагружения (CSR, нормированный цикл). Помимо данных, в функцию потерь добавлен
**физический остаток** дискретизированного уравнения генерации порового давления

    dr/dn ≈ α · CSR · (1 − r),

где α ≥ 0 — скорость накопления, предсказываемая из свойств грунта. Это прямой конкурент
физически-структурированных моделей проекта в парадигме PINN (data loss + PDE/ODE residual).
Реализует контракт ``forward_batch`` / ``compute_loss``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquefaction_ai.models.blocks import ResidualMLP
from liquefaction_ai.training.losses import gaussian_nll, masked_censored_nliq_loss, masked_mean

__all__ = ["PINNBaseline"]


class PINNBaseline(nn.Module):
    """Physics-informed NN: data NLL + остаток ODE порового давления."""

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96, residual_weight: float = 0.2):
        """
        :param static_dim: размерность статических признаков
        :param seq_dim: размерность последовательностных признаков на шаге
        :param hidden_dim: ширина скрытого слоя
        :param residual_weight: вес физического остатка в функции потерь
        """
        super().__init__()
        self.residual_weight = residual_weight
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.trunk = ResidualMLP(hidden_dim + 2, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.r_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        :param batch: словарь батча (поля ``static``, ``seq_in``)
        :return: ``traj_mean``, ``traj_logvar``, ``risk_logit``, ``risk_prob``, ``nliq_pred``, ``alpha``, ``csr``
        """
        seq = batch["seq_in"]
        B, T, _ = seq.shape
        csr = seq[..., 0]
        ncyc = seq[..., 1]
        se = self.static_proj(batch["static"])
        se_exp = se.unsqueeze(1).expand(-1, T, -1)
        inp = torch.cat([se_exp, csr.unsqueeze(-1), ncyc.unsqueeze(-1)], dim=-1)
        h = self.trunk(inp)
        r = torch.sigmoid(self.r_head(h).squeeze(-1))
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-6.0, max=2.0)
        alpha = F.softplus(self.alpha_head(se).squeeze(-1))
        pooled = h.mean(dim=1)
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {"traj_mean": r, "traj_logvar": logvar, "risk_logit": risk_logit,
                "risk_prob": torch.sigmoid(risk_logit), "nliq_pred": nliq_pred,
                "alpha": alpha, "csr": csr}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Data NLL + BCE-риск + Smooth-L1 N_liq + физический остаток ODE + монотонность."""
        out = self.forward_batch(batch)
        r = out["traj_mean"]; csr = out["csr"]; mask = batch["mask"]
        traj_loss = gaussian_nll(r, out["traj_logvar"], batch["r_obs"], mask)
        risk_loss = F.binary_cross_entropy_with_logits(out["risk_logit"], batch["label"])
        nliq_loss = masked_censored_nliq_loss(out["nliq_pred"], batch["n_liq_norm"], batch["label"], batch.get("n_liq_observed"))
        # физический остаток: dr/dn − α·CSR·(1−r) (конечная разность по индексу цикла)
        dr = r[:, 1:] - r[:, :-1]
        rhs = out["alpha"].unsqueeze(1) * csr[:, :-1] * (1.0 - r[:, :-1])
        residual = masked_mean((dr - rhs) ** 2, mask[:, 1:])
        monotonicity = masked_mean(torch.relu(-dr), mask[:, 1:])
        out["loss"] = (traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss
                       + self.residual_weight * residual + 0.05 * monotonicity)
        return out
