"""
Наблюдаемые цели supervision, выводимые из измеренной кривой PPR(N).

Реальный опыт даёт полную траекторию порового давления PPR(N), число циклов до разжижения
и метку разжижения. Из них можно вывести **наблюдаемые** вспомогательные сигналы обучения,
играющие роль deep-supervision и мягкой калибровки риска, но без синтетических латентных
величин:

- ``g_obs`` — мягкий триггер события: сглаженный индикатор достижения порога разжижения
  (PPR ≈ 1) по нарастающему максимуму измеренной кривой. Это классический критерий разжижения,
  применённый к измерению.
- ``risk_proxy`` — мягкая наблюдаемая оценка риска: пиковое измеренное PPR (насколько близко
  образец подошёл к разжижению).

Опционально, если для грунта измерена кривая потенциала разжижения CRR(N) (например, по серии
из 6 образцов), её можно подать как ``crr_obs`` с по-образцовой маской ``crr_obs_mask`` —
тогда внутренняя граница CRR модели обучается по реальному измерению там, где оно доступно.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from liquefaction_ai.config import LIQ_THRESHOLD

__all__ = ["derive_observed_targets"]


def derive_observed_targets(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    liq_threshold: float = LIQ_THRESHOLD,
    kappa: float = 12.0,
) -> Dict[str, np.ndarray]:
    """
    Вывести наблюдаемые вспомогательные цели обучения из измеренной кривой PPR(N).

    Триггер ``g_obs`` строится по нарастающему максимуму измеренного PPR как сглаженный
    индикатор пересечения порога разжижения: g = sigmoid(κ·(cummax(PPR) − thr)); он
    монотонно не убывает (событие, раз наступив, остаётся). Мягкий риск ``risk_proxy`` —
    пиковое измеренное PPR на валидном участке, отсечённое в [0, 1].

    :param r_obs: измеренная траектория PPR, форма (n, seq_len)
    :param valid_mask: маска валидной длины измерений, форма (n, seq_len)
    :param liq_threshold: порог разжижения по PPR (обычно ≈ 0.9…1.0)
    :param kappa: крутизна сглаженного индикатора триггера
    :return: словарь с ``g_obs`` (n, seq_len) и ``risk_proxy`` (n,)
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=np.float64)

    running_max = np.maximum.accumulate(r_obs, axis=1)
    g_obs = 1.0 / (1.0 + np.exp(-kappa * (running_max - liq_threshold)))

    masked = np.where(valid_mask > 0, r_obs, -np.inf)
    peak = masked.max(axis=1)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    risk_proxy = np.clip(peak, 0.0, 1.0)

    return {"g_obs": g_obs.astype(np.float32), "risk_proxy": risk_proxy.astype(np.float32)}
