"""
Базовые (black-box) модели для сравнения со структурированными архитектурами.

Три baseline разной природы, намеренно компактные для исполнимости на обычной машине:
- ``RiskMLP``     — статический MLP-классификатор риска и регрессор N_liq по сводным признакам;
- ``GRUBaseline`` — рекуррентная последовательностная модель с вероятностной головой PPR;
- ``TCNBaseline`` — каузальная временная свёрточная модель без физического слоя.

Все модели реализуют единый контракт ``forward_batch`` и ``compute_loss``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquefaction_ai.models.blocks import CausalTemporalBlock, ResidualMLP
from liquefaction_ai.training.losses import gaussian_nll, masked_mean

__all__ = ["RiskMLP", "GRUBaseline", "TCNBaseline"]


class RiskMLP(nn.Module):
    """
    Статический MLP-базлайн риска разжижения и числа циклов N_liq.

    Использует только статические дескрипторы грунта/нагружения и сводку префикса
    (без полной траектории). Предсказывает логит риска, нормированный N_liq и
    прокси неопределённости.
    """

    def __init__(self, static_dim: int, prefix_dim: int, hidden_dim: int = 128):
        """
        :param static_dim: размерность статических признаков
        :param prefix_dim: размерность сводки префикса
        :param hidden_dim: размерность скрытого представления
        """
        super().__init__()
        self.backbone = ResidualMLP(static_dim + prefix_dim, hidden_dim=hidden_dim, depth=3)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Прямой проход по батчу.

        :param batch: словарь батча (поля ``static``, ``prefix_summary`` и др.)
        :return: словарь выходов: ``risk_logit``, ``risk_prob``, ``nliq_pred``, ``uncertainty``
        """
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
        """
        Вычислить суммарную функцию потерь и выходы по батчу.

        Использует только наблюдаемые сигналы: бинарную метку разжижения и число циклов до
        разжижения (BCE-риск + Smooth-L1 по N_liq).

        :param batch: словарь батча с наблюдаемыми таргетами ``label``/``n_liq_norm``
        :return: словарь выходов с добавленным ключом ``loss``
        """
        outputs = self.forward_batch(batch)
        risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["label"])
        nliq_loss = F.smooth_l1_loss(outputs["nliq_pred"], batch["n_liq_norm"])
        loss = risk_loss + 0.45 * nliq_loss
        outputs["loss"] = loss
        return outputs


class GRUBaseline(nn.Module):
    """
    Рекуррентный (GRU) последовательностный базлайн с вероятностной головой PPR.

    Статические признаки проецируются и конкатенируются с последовательностными
    входами на каждом шаге. Двухслойный GRU предсказывает поэлементное среднее и
    логдисперсию траектории PPR, а из последнего скрытого состояния — риск и N_liq.
    """

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        """
        :param static_dim: размерность статических признаков
        :param seq_dim: размерность последовательностных признаков на шаге
        :param hidden_dim: размерность скрытого состояния GRU
        """
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(input_size=seq_dim + hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2, dropout=0.10)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Прямой проход по батчу.

        :param batch: словарь батча (поля ``static``, ``seq_in`` и др.)
        :return: словарь выходов: ``traj_mean``, ``traj_logvar``, ``risk_logit``, ``risk_prob``, ``nliq_pred``
        """
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
        """
        Вычислить суммарную функцию потерь и выходы по батчу.

        Складывает гауссовскую NLL по траектории PPR, BCE-риск, Smooth-L1 по N_liq
        и штраф гладкости первого порядка.

        :param batch: словарь батча с таргетами и масками
        :return: словарь выходов с добавленным ключом ``loss``
        """
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


class TCNBaseline(nn.Module):
    """
    Каузальный временной свёрточный (TCN) базлайн с вероятностной головой PPR.

    Стек дилатированных каузальных блоков (дилатации 1, 2, 4) расширяет рецептивное
    поле без нарушения каузальности. Предсказывает поэлементные среднее и логдисперсию
    траектории PPR, а также риск и N_liq из последнего временного шага.
    """

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        """
        :param static_dim: размерность статических признаков
        :param seq_dim: размерность последовательностных признаков на шаге
        :param hidden_dim: число каналов скрытых свёрток
        """
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
        """
        Прямой проход по батчу.

        :param batch: словарь батча (поля ``static``, ``seq_in`` и др.)
        :return: словарь выходов: ``traj_mean``, ``traj_logvar``, ``risk_logit``, ``risk_prob``, ``nliq_pred``
        """
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
        """
        Вычислить суммарную функцию потерь и выходы по батчу.

        Складывает гауссовскую NLL по траектории PPR, BCE-риск, Smooth-L1 по N_liq
        и штраф гладкости первого порядка.

        :param batch: словарь батча с таргетами и масками
        :return: словарь выходов с добавленным ключом ``loss``
        """
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
