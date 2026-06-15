"""
Базовые нейросетевые блоки, переиспользуемые архитектурами пакета.

Содержит остаточный MLP-энкодер (общий backbone для всех моделей) и каузальный
временной свёрточный блок с дилатацией (строительный элемент TCN-базлайна).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResidualMLP", "CausalTemporalBlock"]


class ResidualMLP(nn.Module):
    """
    Остаточный многослойный перцептрон с LayerNorm, GELU и dropout.

    Входной линейный слой проецирует признаки в скрытое пространство, далее
    применяется ``depth`` остаточных блоков ``Linear → LayerNorm → GELU → Dropout``.
    Остаточные связи стабилизируют обучение и позволяют наращивать глубину.
    """

    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 3, dropout: float = 0.10):
        """
        :param input_dim: размерность входных признаков
        :param hidden_dim: размерность скрытого представления
        :param depth: число остаточных блоков
        :param dropout: вероятность dropout внутри блоков
        """
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
        """
        Прямой проход энкодера.

        :param x: входной тензор формы (batch, input_dim)
        :return: скрытое представление формы (batch, hidden_dim)
        """
        h = F.gelu(self.input_layer(x))
        for block in self.blocks:
            h = h + block(h)
        return h


class CausalTemporalBlock(nn.Module):
    """
    Каузальный временной свёрточный блок с дилатацией (элемент TCN).

    Две дилатированные свёртки с «обрезкой» (chomp) правого паддинга обеспечивают
    каузальность (выход в момент t зависит только от t и более ранних шагов).
    Остаточная связь с проекцией 1×1 при несовпадении числа каналов ускоряет обучение.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.10):
        """
        :param in_channels: число входных каналов
        :param out_channels: число выходных каналов
        :param kernel_size: размер свёрточного ядра
        :param dilation: коэффициент дилатации (расширения рецептивного поля)
        :param dropout: вероятность dropout между свёртками
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def chomp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Срезать лишний правый паддинг для обеспечения каузальности.

        :param x: тензор формы (batch, channels, time + padding)
        :return: тензор формы (batch, channels, time)
        """
        return x[:, :, : -self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход блока с остаточной связью.

        :param x: входной тензор формы (batch, in_channels, time)
        :return: выходной тензор формы (batch, out_channels, time)
        """
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
