from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 3, dropout: float = 0.10):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.input_layer(x))
        for block in self.blocks:
            h = h + block(h)
        return h


class RiskMLP(nn.Module):
    def __init__(self, static_dim: int, prefix_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = ResidualMLP(static_dim + prefix_dim, hidden_dim=hidden_dim, depth=3)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = torch.cat([batch["static"], batch["prefix_summary"]], dim=-1)
        h = self.backbone(x)
        risk_logit = self.risk_head(h).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(h).squeeze(-1))
        uncertainty = F.softplus(self.uncertainty_head(h).squeeze(-1)) + 1e-3
        return {
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
            "uncertainty": uncertainty,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.forward_batch(batch)
        risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["label"])
        nliq_loss = F.smooth_l1_loss(outputs["nliq_pred"], batch["n_liq_norm"])
        calibration_loss = F.mse_loss(outputs["risk_prob"], batch["risk_true"])
        uncertainty_penalty = F.mse_loss(outputs["uncertainty"], batch["uncertainty_proxy"])
        loss = risk_loss + 0.45 * nliq_loss + 0.20 * calibration_loss + 0.05 * uncertainty_penalty
        outputs["loss"] = loss
        return outputs


class GRUBaseline(nn.Module):
    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(input_size=seq_dim + hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2, dropout=0.10)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        static_embed = self.static_proj(batch["static"]).unsqueeze(1).expand(-1, batch["seq_in"].shape[1], -1)
        x = torch.cat([batch["seq_in"], static_embed], dim=-1)
        h, _ = self.gru(x)
        mean = torch.sigmoid(self.mean_head(h).squeeze(-1))
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-6.0, max=2.0)
        pooled = h[:, -1]
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {
            "traj_mean": mean,
            "traj_logvar": logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from liquefaction_ai.utils.losses import gaussian_nll, masked_mean

        outputs = self.forward_batch(batch)
        traj_loss = gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch["mask"])
        risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["label"])
        nliq_loss = F.smooth_l1_loss(outputs["nliq_pred"], batch["n_liq_norm"])
        smoothness = masked_mean(
            torch.abs(outputs["traj_mean"][:, 1:] - outputs["traj_mean"][:, :-1]), batch["mask"][:, 1:]
        )
        loss = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smoothness
        outputs["loss"] = loss
        return outputs


class CausalTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.10):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def chomp(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.chomp(self.conv1(x))
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.chomp(self.conv2(out))
        out = F.gelu(out)
        out = self.dropout(out)
        residual = self.downsample(x)
        if self.padding > 0:
            residual = residual[:, :, : out.shape[2]]
        return F.gelu(out + residual)


class TCNBaseline(nn.Module):
    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        in_channels = seq_dim + hidden_dim
        self.blocks = nn.Sequential(
            CausalTemporalBlock(in_channels, hidden_dim, kernel_size=3, dilation=1),
            CausalTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            CausalTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
        )
        self.mean_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.logvar_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        static_embed = self.static_proj(batch["static"]).unsqueeze(1).expand(-1, batch["seq_in"].shape[1], -1)
        x = torch.cat([batch["seq_in"], static_embed], dim=-1).transpose(1, 2)
        h = self.blocks(x)
        mean = torch.sigmoid(self.mean_head(h).squeeze(1))
        logvar = torch.clamp(self.logvar_head(h).squeeze(1), min=-6.0, max=2.0)
        pooled = h[:, :, -1]
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {
            "traj_mean": mean,
            "traj_logvar": logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from liquefaction_ai.utils.losses import gaussian_nll, masked_mean

        outputs = self.forward_batch(batch)
        traj_loss = gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch["mask"])
        risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["label"])
        nliq_loss = F.smooth_l1_loss(outputs["nliq_pred"], batch["n_liq_norm"])
        smoothness = masked_mean(
            torch.abs(outputs["traj_mean"][:, 1:] - outputs["traj_mean"][:, :-1]), batch["mask"][:, 1:]
        )
        loss = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smoothness
        outputs["loss"] = loss
        return outputs
