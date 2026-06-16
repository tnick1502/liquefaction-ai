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
import torch.nn.functional as F

__all__ = ["masked_mean", "masked_mse", "masked_mae", "gaussian_nll", "clone_state_dict", "observed_aux_loss"]


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


def observed_aux_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    use_states: bool = True,
    w_g: float = 0.10,
    w_risk: float = 0.10,
    w_crr: float = 0.10,
) -> torch.Tensor:
    """
    Наблюдаемая вспомогательная супервизия, выводимая из измеренной кривой PPR.

    Аналог deep-supervision и калибровки риска, но с **наблюдаемыми** целями (доступными и на
    реальных данных): мягкий триггер ``g_obs`` (момент PPR≈1), мягкий риск ``risk_proxy``
    (пиковое PPR) и, опционально, измеренная граница ``crr_obs`` (с по-образцовой маской
    ``crr_obs_mask``). Все слагаемые подключаются только при наличии соответствующих целей.

    :param outputs: выходы модели (ожидаются ``risk_prob`` и, для физических моделей, ``g``, ``crr``)
    :param batch: словарь батча с наблюдаемыми целями (``risk_proxy``/``g_obs``/``crr_obs``/``mask``)
    :param use_states: применять ли супервизию латентных состояний g и границы CRR
    :param w_g: вес супервизии триггера g
    :param w_risk: вес калибровки риска к наблюдаемому риск-прокси
    :param w_crr: вес супервизии измеренной границы CRR
    :return: скалярный тензор суммарной вспомогательной потери
    """
    device = outputs["risk_prob"].device if "risk_prob" in outputs else outputs["traj_mean"].device
    total = torch.zeros((), device=device)
    if "risk_proxy" in batch and "risk_prob" in outputs:
        total = total + w_risk * F.mse_loss(outputs["risk_prob"], batch["risk_proxy"])
    if use_states and "g_obs" in batch and "g" in outputs:
        total = total + w_g * masked_mse(outputs["g"], batch["g_obs"], batch["mask"])
    if use_states and "crr_obs" in batch and "crr" in outputs:
        mask = batch["mask"]
        per_sample = (((outputs["crr"] - batch["crr_obs"]) ** 2) * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        crr_mask = batch.get("crr_obs_mask")
        if crr_mask is not None:
            total = total + w_crr * (per_sample * crr_mask).sum() / torch.clamp(crr_mask.sum(), min=1.0)
        else:
            total = total + w_crr * per_sample.mean()
    return total


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Создать глубокую копию весов модели на CPU.

    Используется для сохранения лучшего по валидации состояния без удержания
    ссылок на графы вычислений и без привязки к устройству.

    :param model: модель PyTorch
    :return: словарь весов (detached-копии тензоров на CPU)
    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
