"""
Сглаживание траектории порового давления PPR(N) из сырого циклического опыта.

В сыром опыте поровое давление колеблется внутри каждого цикла (квазисинусоида). Для
обучения нужна **гладкая линия PPR(N)** — огибающая по верхним точкам этих колебаний.
Модуль повторяет приём лаборатории (digitrock ``define_max_rude``: пик в каждом цикле) и
дополняет его монотонным сглаживанием:

1. из сырых ``(cycles, PPR)`` извлекаются верхние точки — по одному пику на цикл;
2. по пикам строится **неубывающая** аппроксимация (изотоническая регрессия) — физично,
   т.к. накопленное поровое давление не убывает;
3. результат слегка сглаживается и укладывается на регулярную сетку длины ``seq_len`` с
   маской валидной длины.

Функция :func:`smooth_ppr_trajectory` — единая точка получения гладкой линии PPR для
адаптеров реальных данных (ноутбуки 1.1.3 и 1.1.4).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["extract_upper_envelope", "monotone_smooth", "resample_to_grid", "smooth_ppr_trajectory"]


def extract_upper_envelope(
    cycles: np.ndarray,
    ppr: np.ndarray,
    points_in_cycle: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлечь верхние точки квазисинусоиды PPR — по одному пику на цикл.

    Если задан ``points_in_cycle``, пик берётся в окне фиксированной длины (как
    ``define_max_rude`` в digitrock); иначе — как максимум PPR в пределах каждого целого
    цикла (устойчиво к нерегулярной дискретизации). Пустые циклы заполняются предыдущим
    значением (огибающая не «проваливается»).

    :param cycles: номер цикла в каждой измеренной точке, форма (m,)
    :param ppr: поровое давление в каждой точке, форма (m,)
    :param points_in_cycle: число точек на цикл (опционально, оконный режим)
    :return: кортеж ``(cycle_peaks, ppr_peaks)`` — номера циклов и пики PPR, форма (L,)
    """
    cyc = np.asarray(cycles, dtype=float)
    pp = np.nan_to_num(np.asarray(ppr, dtype=float))
    size = min(cyc.size, pp.size)
    cyc, pp = cyc[:size], pp[:size]
    n_total = int(np.floor(np.nanmax(cyc))) if cyc.size else 0
    n_total = max(n_total, 1)

    if points_in_cycle and points_in_cycle >= 2 and cyc.size >= points_in_cycle:
        cyc_peaks, ppr_peaks = [], []
        for i in range(0, cyc.size, int(points_in_cycle)):
            window = pp[i:i + int(points_in_cycle)]
            if window.size == 0:
                continue
            j = int(np.argmax(window)) + i
            cyc_peaks.append(cyc[j])
            ppr_peaks.append(pp[j])
        cyc_peaks = np.asarray(cyc_peaks, dtype=float)
        ppr_peaks = np.asarray(ppr_peaks, dtype=float)
        order = np.argsort(cyc_peaks)
        return cyc_peaks[order], ppr_peaks[order]

    # Режим по умолчанию: пик PPR в пределах каждого целого цикла
    bins = np.clip(np.ceil(cyc).astype(int), 1, n_total) - 1
    peaks = np.zeros(n_total)
    np.maximum.at(peaks, bins, pp)
    for k in range(1, n_total):           # заполнить пустые циклы предыдущим пиком
        if peaks[k] == 0.0 and peaks[k - 1] > 0.0:
            peaks[k] = peaks[k - 1]
    cyc_peaks = np.arange(1, n_total + 1, dtype=float)
    return cyc_peaks, peaks


def _isotonic_nondecreasing(y: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """Неубывающая изотоническая регрессия (PAVA) с отсечением в [y_min, y_max]."""
    try:
        from sklearn.isotonic import IsotonicRegression

        x = np.arange(len(y), dtype=float)
        fitted = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=True).fit_transform(x, y)
        return np.asarray(fitted, dtype=float)
    except Exception:
        # Запасной алгоритм PAVA без sklearn
        y = np.clip(np.asarray(y, dtype=float), y_min, y_max)
        return np.maximum.accumulate(y)


def monotone_smooth(
    ppr_peaks: np.ndarray,
    smoothing: float = 1.0,
    clip: Tuple[float, float] = (0.0, 1.05),
) -> np.ndarray:
    """
    Построить гладкую неубывающую линию по пикам PPR.

    Сначала пики приводятся к неубывающей огибающей (изотоническая регрессия), затем
    слегка сглаживаются гауссовым ядром и повторно делаются монотонными (cummax).

    :param ppr_peaks: пики PPR по циклам, форма (L,)
    :param smoothing: ширина гауссова сглаживания (σ в циклах); 0 — без сглаживания
    :param clip: пределы значений PPR
    :return: гладкая неубывающая линия PPR, форма (L,)
    """
    y = np.asarray(ppr_peaks, dtype=float)
    if y.size <= 2:
        return np.clip(np.maximum.accumulate(np.nan_to_num(y)), clip[0], clip[1])

    iso = _isotonic_nondecreasing(np.nan_to_num(y), clip[0], clip[1])
    if smoothing and smoothing > 0:
        try:
            from scipy.ndimage import gaussian_filter1d

            iso = gaussian_filter1d(iso, sigma=float(smoothing), mode="nearest")
        except Exception:
            kernel = max(int(round(smoothing)) * 2 + 1, 3)
            pad = kernel // 2
            padded = np.pad(iso, pad, mode="edge")
            iso = np.convolve(padded, np.ones(kernel) / kernel, mode="valid")
        iso = np.maximum.accumulate(iso)        # восстановить монотонность после сглаживания
    return np.clip(iso, clip[0], clip[1])


def resample_to_grid(
    cycle_peaks: np.ndarray,
    ppr_smooth: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Уложить гладкую линию PPR на регулярную сетку длины ``seq_len``.

    Короткий опыт (циклов ≤ ``seq_len``) кладётся в первые узлы, хвост помечается
    невалидным маской; длинный опыт равномерно ресэмплируется на ``seq_len`` узлов.

    :param cycle_peaks: номера циклов пиков, форма (L,)
    :param ppr_smooth: гладкие значения PPR в этих циклах, форма (L,)
    :param seq_len: длина выходной сетки
    :return: кортеж ``(cycles_grid, ppr_grid, valid_mask)`` формы (seq_len,)
    """
    L = len(ppr_smooth)
    grid = np.zeros(seq_len, dtype=np.float32)
    vals = np.zeros(seq_len, dtype=np.float32)
    mask = np.zeros(seq_len, dtype=np.float32)
    if L == 0:
        return grid, vals, mask
    if L <= seq_len:
        grid[:L] = cycle_peaks
        vals[:L] = ppr_smooth
        mask[:L] = 1.0
        grid[L:] = cycle_peaks[-1]
    else:
        idx = np.linspace(0, L - 1, seq_len)
        grid = (np.interp(idx, np.arange(L), cycle_peaks)).astype(np.float32)
        vals = (np.interp(idx, np.arange(L), ppr_smooth)).astype(np.float32)
        mask[:] = 1.0
    return grid, vals, mask


def smooth_ppr_trajectory(
    raw_cycles: np.ndarray,
    raw_ppr: np.ndarray,
    seq_len: int,
    points_in_cycle: Optional[int] = None,
    smoothing: float = 1.0,
    clip: Tuple[float, float] = (0.0, 1.05),
    return_peaks: bool = False,
):
    """
    Полный путь от сырой квазисинусоиды PPR к гладкой линии на сетке ``seq_len``.

    Объединяет три шага: извлечение верхних точек (:func:`extract_upper_envelope`),
    монотонное сглаживание (:func:`monotone_smooth`) и укладку на сетку
    (:func:`resample_to_grid`).

    :param raw_cycles: номера циклов в каждой измеренной точке
    :param raw_ppr: сырое поровое давление в каждой точке
    :param seq_len: длина выходной сетки по циклам
    :param points_in_cycle: число точек на цикл (опционально, оконный режим извлечения пиков)
    :param smoothing: ширина гауссова сглаживания (σ в циклах)
    :param clip: пределы значений PPR
    :param return_peaks: если True — дополнительно вернуть сырые пики ``(cycle_peaks, ppr_peaks)``
    :return: ``(cycles_grid, ppr_grid, valid_mask)`` или, при ``return_peaks``,
             ``(cycles_grid, ppr_grid, valid_mask, cycle_peaks, ppr_peaks)``
    """
    cycle_peaks, ppr_peaks = extract_upper_envelope(raw_cycles, raw_ppr, points_in_cycle)
    ppr_smooth = monotone_smooth(ppr_peaks, smoothing=smoothing, clip=clip)
    grid, vals, mask = resample_to_grid(cycle_peaks, ppr_smooth, seq_len)
    if return_peaks:
        return grid, vals, mask, cycle_peaks, ppr_peaks
    return grid, vals, mask
