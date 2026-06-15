"""
Функции потерь и вспомогательные операции для обучения моделей.

Содержит маскированные варианты ошибок (учитывают валидную длину наблюдения),
гауссовскую отрицательную лог-правдоподобность для вероятностных голов и утилиту
копирования состояния модели для механизма ранней остановки по лучшей валидации.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["masked_mean", "masked_mse", "masked_mae", "gaussian_nll", "clone_state_dict"]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Среднее по элементам с учётом бинарной маски валидности.

    Формула:
        masked_mean = Σ(values · mask) / max(Σ mask, 1)

    :param values: тензор значений
    :param mask: бинарная маска той же формы (1 — валидно, 0 — игнорировать)
    :return: скалярный тензор взвешенного среднего
    """
    return (values * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Маскированная среднеквадратичная ошибка.

    Формула:
        MSE = masked_mean((pred − target)²)

    :param pred: предсказанные значения
    :param target: целевые значения
    :param mask: бинарная маска валидности
    :return: скалярный тензор MSE по валидным элементам
    """
    return masked_mean((pred - target) ** 2, mask)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Маскированная средняя абсолютная ошибка.

    Формула:
        MAE = masked_mean(|pred − target|)

    :param pred: предсказанные значения
    :param target: целевые значения
    :param mask: бинарная маска валидности
    :return: скалярный тензор MAE по валидным элементам
    """
    return masked_mean(torch.abs(pred - target), mask)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Маскированная гауссовская отрицательная лог-правдоподобность.

    Для гетероскедастичной модели с предсказанными средним μ и логарифмом дисперсии
    log σ² минимизируется
        NLL = masked_mean( 0.5·(log σ² + (target − μ)² · exp(−log σ²)) ).

    Логарифм дисперсии отсекается в [−6, 3] для численной устойчивости.

    :param mean: предсказанное среднее μ
    :param logvar: предсказанный логарифм дисперсии log σ²
    :param target: целевые значения
    :param mask: бинарная маска валидности
    :return: скалярный тензор отрицательного лог-правдоподобия
    """
    logvar = torch.clamp(logvar, min=-6.0, max=3.0)
    inv_var = torch.exp(-logvar)
    return masked_mean(0.5 * (logvar + (target - mean) ** 2 * inv_var), mask)


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Создать глубокую копию весов модели на CPU.

    Используется для сохранения лучшего по валидации состояния без удержания
    ссылок на графы вычислений и без привязки к устройству.

    :param model: модель PyTorch
    :return: словарь весов (detached-копии тензоров на CPU)
    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
