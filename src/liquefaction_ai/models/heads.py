"""
Обучаемые выходные головы для физически-структурированных моделей.

Вместо фиксированной линейной формулы риска физические модели используют обучаемую
голову, которая агрегирует контекст и сводные признаки смоделированной физики
(пиковые и финальные значения PPR, повреждения, триггера и оценки N_liq). Это даёт
калиброванную вероятность разжижения. Дополнительно гетероскедастичная голова
предсказывает логарифм дисперсии PPR по шагам, делая интервалы неопределённости
зависящими от состояния.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["physics_summary", "RiskHead", "SeqLogvarHead"]

PHYSICS_SUMMARY_DIM = 8
"""Размерность вектора сводных признаков физики (см. :func:`physics_summary`)."""


def physics_summary(r: torch.Tensor, z: torch.Tensor, g: torch.Tensor, nliq_norm: torch.Tensor) -> torch.Tensor:
    """
    Сформировать сводные признаки смоделированной физической динамики.

    Признаки: пик, финальное и среднее значение PPR; пик и финальное значение скрытого
    повреждения z; пик и среднее значение триггера g; нормированная оценка N_liq.
    Эти величины — компактное и информативное представление траектории для головы риска.

    :param r: траектория PPR, форма (batch, seq_len)
    :param z: траектория скрытого повреждения, форма (batch, seq_len)
    :param g: траектория триггера события, форма (batch, seq_len)
    :param nliq_norm: нормированная оценка N_liq, форма (batch,)
    :return: тензор сводных признаков, форма (batch, 8)
    """
    return torch.stack(
        [
            r.amax(dim=1), r[:, -1], r.mean(dim=1),
            z.amax(dim=1), z[:, -1],
            g.amax(dim=1), g.mean(dim=1),
            nliq_norm,
        ],
        dim=-1,
    )


class RiskHead(nn.Module):
    """
    Обучаемая голова оценки риска разжижения.

    Принимает закодированный контекст и сводные признаки физики, возвращает логит риска.
    Заменяет фиксированную линейную комбинацию пиков состояний, что улучшает калибровку.
    """

    def __init__(self, context_dim: int, summary_dim: int = PHYSICS_SUMMARY_DIM, hidden_dim: int = 64):
        """
        :param context_dim: размерность закодированного контекста
        :param summary_dim: размерность сводных признаков физики
        :param hidden_dim: размерность скрытого слоя головы
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Нулевая инициализация выхода: голова стартует как остаток ≈ 0 к физическому prior,
        # сохраняя сильную физическую инициализацию риска и обучая лишь поправку калибровки.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """
        Вычислить остаточную поправку к логиту риска (стартует ≈ 0).

        :param context: закодированный контекст, форма (batch, context_dim)
        :param summary: сводные признаки физики, форма (batch, summary_dim)
        :return: остаточная поправка логита риска, форма (batch,)
        """
        return self.net(torch.cat([context, summary], dim=-1)).squeeze(-1)


class SeqLogvarHead(nn.Module):
    """
    Гетероскедастичная голова логарифма дисперсии траектории PPR.

    По контексту предсказывает логарифм дисперсии для каждого шага последовательности,
    что делает ширину интервала неопределённости зависящей от сценария и шага.
    """

    def __init__(self, context_dim: int, seq_len: int, hidden_dim: int = 64,
                 logvar_min: float = -8.0, logvar_max: float = -1.0, init_logvar: float = -4.0):
        """
        :param context_dim: размерность закодированного контекста
        :param seq_len: длина последовательности
        :param hidden_dim: размерность скрытого слоя
        :param logvar_min: нижняя отсечка логарифма дисперсии
        :param logvar_max: верхняя отсечка логарифма дисперсии
        :param init_logvar: стартовый (постоянный) логарифм дисперсии для устойчивости NLL
        """
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_len),
        )
        # Старт как гомоскедастичная оценка ~init_logvar: голова учит лишь гетероскед. поправку.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_logvar)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Предсказать логарифм дисперсии по шагам.

        :param context: закодированный контекст, форма (batch, context_dim)
        :return: логарифм дисперсии, форма (batch, seq_len), отсечён в допустимый диапазон
        """
        return torch.clamp(self.net(context), min=self.logvar_min, max=self.logvar_max)
