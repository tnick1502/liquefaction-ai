from typing import Union

import numpy as np


ArrayLike = Union[np.ndarray, list, tuple]


def compute_r2(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """
    Расчёт коэффициента детерминации R².

    Формула:
        R² = 1 - SS_res / SS_tot

    где SS_res = Σ(y_i - ŷ_i)², SS_tot = Σ(y_i - ȳ)².

    :param y_true: массив истинных (наблюдаемых) значений
    :param y_pred: массив предсказанных (модельных) значений
    :return: значение R² (float)
    """
    y_t: np.ndarray = np.asarray(y_true, dtype=np.float64)
    y_p: np.ndarray = np.asarray(y_pred, dtype=np.float64)

    ss_res: float = float(np.sum((y_t - y_p) ** 2))
    ss_tot: float = float(np.sum((y_t - np.mean(y_t)) ** 2))

    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0

    return 1.0 - ss_res / ss_tot


def compute_mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """
    Расчёт среднеквадратичной ошибки MSE.

    Формула:
        MSE = (1 / n) * Σ(y_i - ŷ_i)²

    :param y_true: массив истинных (наблюдаемых) значений
    :param y_pred: массив предсказанных (модельных) значений
    :return: значение MSE (float)
    """
    y_t: np.ndarray = np.asarray(y_true, dtype=np.float64)
    y_p: np.ndarray = np.asarray(y_pred, dtype=np.float64)

    return float(np.mean((y_t - y_p) ** 2))


def compute_wape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """
    Расчёт взвешенной абсолютной процентной ошибки WAPE.

    Формула:
        WAPE = Σ|y_i - ŷ_i| / Σ|y_i|

    :param y_true: массив истинных (наблюдаемых) значений
    :param y_pred: массив предсказанных (модельных) значений
    :return: значение WAPE (float)
    """
    y_t: np.ndarray = np.asarray(y_true, dtype=np.float64)
    y_p: np.ndarray = np.asarray(y_pred, dtype=np.float64)

    abs_sum: float = float(np.sum(np.abs(y_t)))

    if np.isclose(abs_sum, 0.0):
        raise ValueError("Сумма абсолютных значений y_true равна 0 — WAPE не определён.")

    return float(np.sum(np.abs(y_t - y_p)) / abs_sum)
