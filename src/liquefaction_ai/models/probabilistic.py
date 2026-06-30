"""
Вероятностные baseline: глубокая модель пространства состояний и условные нормализующие потоки.

- :class:`DeepStateBaseline` — рекуррентная модель пространства состояний (в духе DeepState/DeepAR):
  GRU предсказывает приращения уровня латентного состояния, уровень накапливается (случайное
  блуждание), наблюдение PPR = sigmoid(уровень); вероятностная (гауссова) эмиссия по шагам.
- :class:`RealNVPFlow` — условный нормализующий поток над траекторией PPR (в logit-пространстве)
  с аффинными coupling-слоями (RealNVP, Dinh et al., 2017). Обучается точным максимумом
  правдоподобия; предсказание — среднее по сэмплам.
- :class:`NeuralSplineFlow` — то же, но с монотонными рационально-квадратичными сплайн-coupling
  слоями (Neural Spline Flows, Durkan et al., 2019) — более гибкое преобразование.

Потоки изолируют вклад гибкой плотности отдельно от ODE/физики (прямое сравнение с flow-частью
DPI-Flow). Все модели реализуют контракт ``forward_batch`` / ``compute_loss``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquefaction_ai.training.losses import (gaussian_nll, masked_bce_with_logits,
                                             masked_censored_nliq_loss, masked_mean,
                                             nliq_censor_mask, risk_observation_mask)

__all__ = ["DeepStateBaseline", "RealNVPFlow", "NeuralSplineFlow"]

_EPS = 1e-4


class DeepStateBaseline(nn.Module):
    """Глубокая модель пространства состояний: GRU → приращения уровня → sigmoid-наблюдение PPR."""

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(input_size=seq_dim + hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2, dropout=0.10)
        self.delta_head = nn.Linear(hidden_dim, 1)        # приращение уровня (≥0 через softplus)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.level0_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        se = self.static_proj(batch["static"])
        se_exp = se.unsqueeze(1).expand(-1, batch["seq_in"].shape[1], -1)
        h, _ = self.gru(torch.cat([batch["seq_in"], se_exp], dim=-1))
        delta = F.softplus(self.delta_head(h).squeeze(-1))          # неотрицательные приращения
        level0 = self.level0_head(se)                              # (B,1)
        level = level0 + torch.cumsum(delta, dim=1)                 # состояние-блуждание
        traj_mean = torch.sigmoid(level)
        traj_logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-6.0, max=2.0)
        pooled = h[:, -1]
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {"traj_mean": traj_mean, "traj_logvar": traj_logvar, "risk_logit": risk_logit,
                "risk_prob": torch.sigmoid(risk_logit), "nliq_pred": nliq_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.forward_batch(batch)
        traj_loss = gaussian_nll(out["traj_mean"], out["traj_logvar"], batch["r_obs"], batch["mask"])
        risk_loss = masked_bce_with_logits(out["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(out["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        out["loss"] = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss
        return out


# ----------------------- условные нормализующие потоки над траекторией -----------------------

def _ff_fill_logit(r_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Заполнить невалидный хвост последним валидным значением (PPR монотонна → cummax) и в logit."""
    filled = torch.cummax(r_obs * (mask > 0), dim=1).values
    filled = torch.clamp(filled, _EPS, 1.0 - _EPS)
    return torch.log(filled) - torch.log1p(-filled)


class _CouplingNet(nn.Module):
    """Условная сеть параметров coupling-слоя: вход [masked_x, context] → выход params·D."""

    def __init__(self, dim: int, ctx_dim: int, out_per_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim * out_per_dim),
        )
        self.dim = dim
        self.out_per_dim = out_per_dim

    def forward(self, x_masked: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([x_masked, ctx], dim=-1))
        return out.view(x_masked.shape[0], self.dim, self.out_per_dim)


def _rqs(x, w, h, d, inverse, bound=4.0, min_bin=1e-3, min_deriv=1e-3):
    """Монотонный рационально-квадратичный сплайн (Durkan 2019) с линейными хвостами вне [-bound,bound]."""
    K = w.shape[-1]
    inside = (x >= -bound) & (x <= bound)
    out = x.clone()
    logabsdet = torch.zeros_like(x)
    if not inside.any():
        return out, logabsdet

    xi = x[inside]
    wi = min_bin + (1 - min_bin * K) * torch.softmax(w[inside], dim=-1)
    hi = min_bin + (1 - min_bin * K) * torch.softmax(h[inside], dim=-1)
    di = min_deriv + F.softplus(d[inside])                       # (n, K-1) внутренние производные
    dd = F.pad(di, (1, 1), value=1.0)                            # хвостовые производные = 1 (линейные хвосты)

    cumw = F.pad(torch.cumsum(wi, dim=-1), (1, 0)) * 2 * bound - bound   # (n, K+1) узлы по x
    cumh = F.pad(torch.cumsum(hi, dim=-1), (1, 0)) * 2 * bound - bound   # (n, K+1) узлы по y

    search = cumh if inverse else cumw
    idx = (torch.searchsorted(search, xi.unsqueeze(-1).contiguous(), right=True) - 1).clamp(0, K - 1)

    xk = cumw.gather(-1, idx).squeeze(-1); xk1 = cumw.gather(-1, idx + 1).squeeze(-1)
    yk = cumh.gather(-1, idx).squeeze(-1); yk1 = cumh.gather(-1, idx + 1).squeeze(-1)
    dk = dd.gather(-1, idx).squeeze(-1); dk1 = dd.gather(-1, idx + 1).squeeze(-1)
    wbin = (xk1 - xk).clamp_min(1e-6)
    hbin = (yk1 - yk)
    s = hbin / wbin

    if not inverse:
        xrel = ((xi - xk) / wbin).clamp(0, 1)
        num = hbin * (s * xrel ** 2 + dk * xrel * (1 - xrel))
        den = s + (dk1 + dk - 2 * s) * xrel * (1 - xrel)
        out[inside] = yk + num / den
        deriv = (s ** 2 * (dk1 * xrel ** 2 + 2 * s * xrel * (1 - xrel) + dk * (1 - xrel) ** 2)) / den ** 2
        logabsdet[inside] = torch.log(deriv.clamp_min(1e-8))
    else:
        yrel = xi - yk
        a = hbin * (s - dk) + yrel * (dk1 + dk - 2 * s)
        b = hbin * dk - yrel * (dk1 + dk - 2 * s)
        c = -s * yrel
        disc = (b ** 2 - 4 * a * c).clamp_min(1e-8)
        xrel = (2 * c / (-b - torch.sqrt(disc))).clamp(0, 1)
        out[inside] = xrel * wbin + xk
        den = s + (dk1 + dk - 2 * s) * xrel * (1 - xrel)
        deriv = (s ** 2 * (dk1 * xrel ** 2 + 2 * s * xrel * (1 - xrel) + dk * (1 - xrel) ** 2)) / den ** 2
        logabsdet[inside] = -torch.log(deriv.clamp_min(1e-8))
    return out, logabsdet


class _ConditionalTrajectoryFlow(nn.Module):
    """Условный нормализующий поток над траекторией PPR (logit-пространство). coupling: 'affine'|'spline'."""

    def __init__(self, dim: int, static_dim: int, prefix_dim: int, coupling: str = "affine",
                 n_layers: int = 6, n_bins: int = 8, ctx_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.coupling = coupling
        self.n_bins = n_bins
        self.ctx_net = nn.Sequential(nn.Linear(static_dim + prefix_dim, ctx_dim), nn.GELU(), nn.Linear(ctx_dim, ctx_dim))
        masks = []
        for i in range(n_layers):
            m = torch.zeros(dim); m[i % 2::2] = 1.0
            masks.append(m)
        self.register_buffer("masks", torch.stack(masks))
        out_per_dim = 2 if coupling == "affine" else 3 * n_bins + 1
        self.nets = nn.ModuleList([_CouplingNet(dim, ctx_dim, out_per_dim) for _ in range(n_layers)])
        # головы риска/N_liq из контекста (предсказание без таргетов)
        self.risk_head = nn.Linear(ctx_dim, 1)
        self.nliq_head = nn.Linear(ctx_dim, 1)

    def _ctx(self, batch):
        return self.ctx_net(torch.cat([batch["static"], batch["prefix_summary"]], dim=-1))

    def _layer(self, x, ctx, net, mask, inverse):
        mb = mask.unsqueeze(0)
        x_masked = x * mb
        params = net(x_masked, ctx)
        if self.coupling == "affine":
            s = torch.tanh(params[..., 0]); t = params[..., 1]
            if not inverse:
                y = x_masked + (1 - mb) * (x * torch.exp(s) + t)
                ld = ((1 - mb) * s).sum(-1)
            else:
                y = x_masked + (1 - mb) * ((x - t) * torch.exp(-s))
                ld = (-(1 - mb) * s).sum(-1)
            return y, ld
        else:
            K = self.n_bins
            w, h, d = params[..., :K], params[..., K:2 * K], params[..., 2 * K:]
            y_tr, ld_tr = _rqs(x, w, h, d, inverse=inverse)
            y = x_masked + (1 - mb) * y_tr
            ld = ((1 - mb) * ld_tr).sum(-1)
            return y, ld

    def log_prob(self, x, ctx):
        z = x; logdet = torch.zeros(x.shape[0], device=x.device)
        for i in range(len(self.nets)):
            z, ld = self._layer(z, ctx, self.nets[i], self.masks[i], inverse=True)
            logdet = logdet + ld
        base = -0.5 * (z ** 2 + math.log(2 * math.pi))
        return base.sum(-1) + logdet

    def sample(self, ctx, K=32):
        B = ctx.shape[0]
        samples = []
        for _ in range(K):
            z = torch.randn(B, self.dim, device=ctx.device)
            x = z
            for i in reversed(range(len(self.nets))):
                x, _ = self._layer(x, ctx, self.nets[i], self.masks[i], inverse=False)
            samples.append(torch.sigmoid(x))
        s = torch.stack(samples, 0)
        return s.mean(0), s.var(0).clamp_min(1e-6)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = self._ctx(batch)
        traj_mean, traj_var = self.sample(ctx, K=24)
        risk_logit = self.risk_head(ctx).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(ctx).squeeze(-1))
        return {"traj_mean": traj_mean, "traj_logvar": torch.log(traj_var),
                "risk_logit": risk_logit, "risk_prob": torch.sigmoid(risk_logit), "nliq_pred": nliq_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = self._ctx(batch)
        y = _ff_fill_logit(batch["r_obs"], batch["mask"])
        nll = -self.log_prob(y, ctx).mean() / self.dim
        risk_logit = self.risk_head(ctx).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(ctx).squeeze(-1))
        risk_loss = masked_bce_with_logits(risk_logit, batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(nliq_pred, batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        loss = nll + 0.35 * risk_loss + 0.25 * nliq_loss
        return {"loss": loss, "risk_logit": risk_logit, "risk_prob": torch.sigmoid(risk_logit),
                "nliq_pred": nliq_pred, "nll": nll}


class RealNVPFlow(_ConditionalTrajectoryFlow):
    """Условный RealNVP над траекторией PPR (аффинные coupling-слои)."""

    def __init__(self, static_dim: int, prefix_dim: int, seq_len: int, n_layers: int = 6):
        super().__init__(seq_len, static_dim, prefix_dim, coupling="affine", n_layers=n_layers)


class NeuralSplineFlow(_ConditionalTrajectoryFlow):
    """Условный Neural Spline Flow над траекторией PPR (рационально-квадратичные сплайны)."""

    def __init__(self, static_dim: int, prefix_dim: int, seq_len: int, n_layers: int = 5, n_bins: int = 8):
        super().__init__(seq_len, static_dim, prefix_dim, coupling="spline", n_layers=n_layers, n_bins=n_bins)
