"""
Абляционные модели для изоляции вклада компонентов DPI-Flow (ODE + нормализующий flow + физика).

Три варианта для честного сравнения «по компонентам», все с единым контрактом
``forward_batch`` / ``compute_loss`` (как в :mod:`liquefaction_ai.models.baselines`):

- :class:`NeuralODENoPhysics` — латентный Neural ODE: обучаемая правая часть dz/dt = f(z, CSR, t),
  интегрируемая по нормированной сетке циклов (Эйлер). PPR(N) = sigmoid(декодер(z_t)). Никаких
  физических связей/приоров и физической супервизии — чистая нейро-динамика («ODE без физики»).
- :class:`FlowNoODE` — энкодер→гауссов латент→условный аффинный нормализующий flow→θ, далее
  обучаемый MLP-декодер θ→PPR(N) (без интегрирования ODE). Вероятностный, с KL («flow без ODE»).
- «ODE без flow» отдельным классом не реализуется: это ``DPIFlow(use_flow=False)`` — отключение
  нормализующего flow при сохранении аналитического ODE-слоя.

Эти варианты показывают, что именно даёт каждый блок: интегрирование динамики (ODE),
гибкое апостериорное распределение параметров (flow) и физические ограничения.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquefaction_ai.models.blocks import ResidualMLP
from liquefaction_ai.models.dpi_flow import ConditionalAffineFlow
from liquefaction_ai.training.losses import (gaussian_nll, masked_bce_with_logits,
                                             masked_censored_nliq_loss, masked_mean,
                                             nliq_censor_mask, risk_observation_mask)

__all__ = ["NeuralODENoPhysics", "FlowNoODE"]


class NeuralODENoPhysics(nn.Module):
    """
    Латентный Neural ODE без физических ограничений.

    Энкодер из статических и префиксных признаков задаёт начальное латентное состояние z0.
    Обучаемая правая часть ``f(z, CSR_t, log_t)`` интегрируется явным методом Эйлера по
    нормированным приращениям циклов; на каждом шаге PPR_t = sigmoid(декодер(z_t)). Риск и
    N_liq читаются из агрегированного латента. Физических приоров и физической супервизии нет.
    """

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ResidualMLP(static_dim, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.z0_head = nn.Linear(hidden_dim, latent_dim)
        # правая часть ODE: вход [z, CSR, log_cycle_norm] -> dz
        self.f = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.ppr_head = nn.Linear(latent_dim, 1)
        self.logvar_head = nn.Linear(latent_dim, 1)
        self.risk_head = nn.Linear(latent_dim, 1)
        self.nliq_head = nn.Linear(latent_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq = batch["seq_in"]                       # (B, T, C): CSR, log_cycle_norm, delta_cycle_norm, ...
        B, T, _ = seq.shape
        csr = seq[..., 0]
        log_cyc = seq[..., 1]
        dt = seq[..., 2].clamp(min=0.0)             # delta_cycle_norm как шаг интегрирования
        z = self.z0_head(self.encoder(batch["static"]))
        means, logvars = [], []
        for t in range(T):
            drive = torch.stack([csr[:, t], log_cyc[:, t]], dim=-1)
            dz = self.f(torch.cat([z, drive], dim=-1))
            z = z + dt[:, t:t + 1] * dz             # шаг Эйлера
            means.append(torch.sigmoid(self.ppr_head(z).squeeze(-1)))
            logvars.append(torch.clamp(self.logvar_head(z).squeeze(-1), min=-6.0, max=2.0))
        traj_mean = torch.stack(means, dim=1)
        traj_logvar = torch.stack(logvars, dim=1)
        pooled = z
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {
            "traj_mean": traj_mean, "traj_logvar": traj_logvar,
            "risk_logit": risk_logit, "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.forward_batch(batch)
        traj_loss = gaussian_nll(out["traj_mean"], out["traj_logvar"], batch["r_obs"], batch["mask"])
        risk_loss = masked_bce_with_logits(out["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(out["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        smooth = masked_mean(torch.abs(out["traj_mean"][:, 1:] - out["traj_mean"][:, :-1]), batch["mask"][:, 1:])
        out["loss"] = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smooth
        return out


class FlowNoODE(nn.Module):
    """
    Нормализующий flow без ODE-интегрирования.

    Энкодер контекста (статические + префикс) задаёт гауссов латент (μ, logσ²); условный
    аффинный flow преобразует его в θ; обучаемый MLP-декодер отображает θ напрямую в кривую
    PPR(N) (без интегрирования динамики). Вероятностная модель с KL-регуляризацией — изолирует
    вклад гибкого апостериорного распределения параметров отдельно от ODE.
    """

    def __init__(self, static_dim: int, prefix_dim: int, seq_len: int, theta_dim: int = 31,
                 hidden_dim: int = 128, probabilistic: bool = True):
        super().__init__()
        self.theta_dim = theta_dim
        self.seq_len = seq_len
        self.probabilistic = probabilistic
        ctx = static_dim + prefix_dim
        self.encoder = ResidualMLP(ctx, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.mu_head = nn.Linear(hidden_dim, theta_dim)
        self.logvar_head = nn.Linear(hidden_dim, theta_dim)
        self.flow = ConditionalAffineFlow(theta_dim, hidden_dim)
        self.decoder = ResidualMLP(theta_dim + hidden_dim, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.traj_head = nn.Linear(hidden_dim, seq_len)
        self.traj_logvar_head = nn.Linear(hidden_dim, seq_len)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = torch.cat([batch["static"], batch["prefix_summary"]], dim=-1)
        enc = self.encoder(ctx)
        mu = self.mu_head(enc)
        raw_logvar = torch.clamp(self.logvar_head(enc), min=-5.0, max=3.0)
        if self.probabilistic:
            eps = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
            latent = mu + torch.exp(0.5 * raw_logvar) * eps
        else:
            latent = mu
            raw_logvar = torch.zeros_like(mu)
        theta = self.flow(latent, enc)
        dec = self.decoder(torch.cat([theta, enc], dim=-1))
        traj_mean = torch.sigmoid(self.traj_head(dec))
        traj_logvar = torch.clamp(self.traj_logvar_head(dec), min=-6.0, max=2.0)
        risk_logit = self.risk_head(dec).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(dec).squeeze(-1))
        kl = 0.5 * (torch.exp(raw_logvar) + mu.pow(2) - 1.0 - raw_logvar).mean(dim=1)
        return {
            "traj_mean": traj_mean, "traj_logvar": traj_logvar,
            "risk_logit": risk_logit, "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred, "kl": kl,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.forward_batch(batch)
        traj_loss = gaussian_nll(out["traj_mean"], out["traj_logvar"], batch["r_obs"], batch["mask"])
        risk_loss = masked_bce_with_logits(out["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(out["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        kl_loss = out["kl"].mean() if self.probabilistic else torch.zeros((), device=out["traj_mean"].device)
        smooth = masked_mean(torch.abs(out["traj_mean"][:, 1:] - out["traj_mean"][:, :-1]), batch["mask"][:, 1:])
        out["loss"] = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smooth + 0.001 * kl_loss
        return out
