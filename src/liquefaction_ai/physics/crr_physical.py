"""
Физическая модель циклической сопротивляемости CRR(N) (перенос из model.py).

Кривая сопротивления записывается в гиперболически-степенной форме
    CRR(N) = β / N^(1 − α),
где параметры α и β выводятся не напрямую, а из физико-механических свойств грунта
через цепочку интерпретируемых поправок к базовому уровню CRR при опорном числе циклов
(обычно N_ref = 15):

    CRR_ref = CRR15_density(Dr) · Π factor_i,
    s = 1 − α  — показатель циклической деградации (наклон в логарифмической шкале),
    β = CRR_ref · N_ref^s.

Поправочные множители отражают вклад плотности, содержания и пластичности мелких частиц,
показателя текучести, органики, бытового давления (K_σ), переуплотнения (OCR), скорости
поперечных волн Vs1, цементации/старения, начального статического сдвига (K_α), режима
нагружения, неполного водонасыщения, демпфирования, частоты и анизотропии K0.

Реализация векторизована (numpy) для применения ко всей популяции грунтов сразу и
повторяет логику исходного модуля model.py.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = [
    "RESPONSE_TYPES",
    "compute_crr_components",
    "crr_curve",
]

RESPONSE_TYPES = ("contractive", "transitional", "dilative", "plastic")
"""Типы циклического отклика грунта."""


def _sigmoid(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    """
    Гладкий логистический переход.

    :param x: входной массив
    :param center: центр перехода
    :param scale: ширина перехода (по модулю, > 0)
    :return: значения логистической функции в (0, 1)
    """
    scale = max(abs(scale), 1e-9)
    z = np.clip((x - center) / scale, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _base_crr15_from_density(dr: np.ndarray) -> np.ndarray:
    """
    Базовый CRR при 15 циклах для чистого нецементированного песка.

    Формула: CRR15 = clip(0.055 + 0.53·Dr^1.65, 0.06, 0.62).

    :param dr: относительная плотность Dr, доли единицы
    :return: базовый CRR15 до физических поправок
    """
    return np.clip(0.055 + 0.53 * np.power(np.clip(dr, 0.0, 1.0), 1.65), 0.06, 0.62)


def _fines_plasticity_factor(fc: np.ndarray, ip: np.ndarray, clay: np.ndarray) -> np.ndarray:
    """
    Поправка на содержание и пластичность мелких частиц.

    :param fc: содержание мелких частиц (< 0.075 мм), %
    :param ip: число пластичности Ip, %
    :param clay: содержание глинистых частиц (< 0.002 мм), %
    :return: множитель к CRR_ref
    """
    fc = np.clip(fc, 0.0, 100.0)
    ip = np.maximum(ip, 0.0)
    clay = np.clip(clay, 0.0, 100.0)

    nonplastic_valley = np.exp(-((fc - 22.0) / 18.0) ** 2)
    val_lowplastic = 1.0 - 0.22 * nonplastic_valley + 0.04 * (fc / 100.0)
    val_midplastic = 0.98 + 0.006 * (ip - 7.0) * np.minimum(fc, 45.0) / 35.0
    val_highplastic = 1.05 + 0.010 * (ip - 15.0) * np.minimum(fc, 60.0) / 60.0

    factor = np.select(
        [fc <= 5.0, ip < 7.0, ip < 15.0],
        [np.ones_like(fc), val_lowplastic, val_midplastic],
        default=val_highplastic,
    )
    factor = np.where((clay > 20.0) & (ip < 7.0), factor * 0.92, factor)
    factor = np.where(
        (clay > 20.0) & (ip >= 12.0),
        factor * (1.0 + 0.08 * np.minimum((clay - 20.0) / 40.0, 1.0)),
        factor,
    )
    return np.clip(factor, 0.70, 1.38)


def _liquidity_factor(il: np.ndarray) -> np.ndarray:
    """
    Поправка на показатель текучести Il.

    :param il: показатель текучести Il
    :return: множитель к CRR_ref
    """
    softening = np.exp(-0.36 * np.clip(il, 0.0, 1.8))
    hardening = 1.0 + 0.12 * np.clip(-il, 0.0, 1.2)
    return np.clip(softening * hardening, 0.52, 1.16)


def _organic_factor(ir: np.ndarray) -> np.ndarray:
    """
    Поправка на содержание органического вещества Ir.

    :param ir: содержание органики, %
    :return: множитель к CRR_ref
    """
    return np.clip(np.exp(-0.0075 * np.maximum(ir, 0.0)), 0.55, 1.0)


def _overburden_factor(sigma_eff: np.ndarray, dr: np.ndarray) -> np.ndarray:
    """
    Поправка K_σ на бытовое (эффективное) давление.

    :param sigma_eff: эффективное напряжение, кПа
    :param dr: относительная плотность Dr, доли единицы
    :return: множитель к CRR_ref
    """
    sigma = np.maximum(sigma_eff, 5.0)
    pa = 100.0
    n160 = np.clip(2.0 + 46.0 * np.power(dr, 2.0), 2.0, 45.0)
    denom = np.maximum(18.9 - 2.55 * np.sqrt(n160), 3.5)
    c_sigma = np.minimum(1.0 / denom, 0.30)
    k_sigma = 1.0 - c_sigma * np.log(sigma / pa)
    return np.clip(k_sigma, 0.65, 1.10)


def _ocr_factor(ocr: np.ndarray) -> np.ndarray:
    """
    Поправка на коэффициент переуплотнения OCR.

    :param ocr: коэффициент переуплотнения
    :return: множитель к CRR_ref
    """
    return np.clip(np.power(np.maximum(ocr, 0.4), 0.16), 0.85, 1.35)


def _vs_factor(vs1: np.ndarray) -> np.ndarray:
    """
    Поправка по скорректированной скорости поперечных волн Vs1.

    :param vs1: скорость поперечных волн, скорректированная на давление, м/с
    :return: множитель к CRR_ref
    """
    return np.clip(0.74 + 0.58 * _sigmoid(vs1, center=170.0, scale=34.0), 0.76, 1.30)


def _cementation_aging_factor(cementation: np.ndarray, aging_years: np.ndarray) -> np.ndarray:
    """
    Поправка на цементацию и возраст (старение) грунта.

    :param cementation: индекс цементации (0…3)
    :param aging_years: возраст грунта, лет
    :return: множитель к CRR_ref
    """
    cementation = np.clip(cementation, 0.0, 3.0)
    aging = np.maximum(aging_years, 0.0)
    return np.clip(1.0 + 0.18 * cementation + 0.055 * np.log10(1.0 + aging), 1.0, 1.55)


def _static_shear_factor(a_static: np.ndarray, response_idx: np.ndarray, dr: np.ndarray) -> np.ndarray:
    """
    Поправка K_α на начальный статический сдвиг.

    :param a_static: коэффициент начального статического сдвига α_static
    :param response_idx: индексы типа отклика (0 contr, 1 trans, 2 dil, 3 plastic)
    :param dr: относительная плотность Dr, доли единицы
    :return: множитель к CRR_ref
    """
    a = np.clip(a_static, 0.0, 0.45)
    contractive = 1.0 - (1.85 - 0.75 * dr) * a
    dilative = 1.0 + 1.15 * a * np.clip((dr - 0.50) / 0.45, 0.0, 1.0)
    plastic = 1.0 - 0.35 * a
    transitional = 1.0 - 0.55 * a + 0.35 * a * dr
    factor = np.select(
        [response_idx == 0, response_idx == 2, response_idx == 3],
        [contractive, dilative, plastic],
        default=transitional,
    )
    factor = np.where(a <= 0.0, 1.0, factor)
    return np.clip(factor, 0.35, 1.35)


def _saturation_factor(sr: np.ndarray, b_value: np.ndarray) -> np.ndarray:
    """
    Поправка на неполное водонасыщение.

    :param sr: степень водонасыщения Sr, доли единицы
    :param b_value: параметр Скемптона B, доли единицы
    :return: множитель к CRR_ref
    """
    factor = (1.0 + 0.20 * np.clip(0.95 - b_value, 0.0, 0.35) / 0.35) * (
        1.0 + 0.12 * np.clip(0.98 - sr, 0.0, 0.30) / 0.30
    )
    return np.clip(factor, 1.0, 1.18)


def _damping_factor(damping_percent: np.ndarray) -> np.ndarray:
    """
    Поправка на начальное демпфирование материала.

    :param damping_percent: коэффициент демпфирования, %
    :return: множитель к CRR_ref
    """
    low = 1.0 + 0.018 * (damping_percent - 5.0) / 5.0
    high = 1.01 * np.exp(-0.018 * (damping_percent - 8.0))
    return np.clip(np.where(damping_percent <= 8.0, low, high), 0.72, 1.06)


def _frequency_factor(frequency: np.ndarray, ip: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """
    Мягкая поправка на частоту циклического нагружения.

    :param frequency: частота нагружения, Гц
    :param ip: число пластичности Ip, %
    :param fc: содержание мелких частиц, %
    :return: множитель к CRR_ref
    """
    f = np.maximum(frequency, 0.02)
    weight = _sigmoid(ip, center=8.0, scale=4.0) * _sigmoid(fc, center=25.0, scale=12.0)
    return np.clip(1.0 + 0.030 * np.log10(f / 1.0) * weight, 0.92, 1.12)


def _infer_response(dr, fc, ip, il, ocr) -> tuple:
    """
    Определить тип циклического отклика и прокси-показатель состояния.

    :param dr: относительная плотность Dr
    :param fc: содержание мелких частиц, %
    :param ip: число пластичности Ip, %
    :param il: показатель текучести Il
    :param ocr: коэффициент переуплотнения
    :return: кортеж (индексы типа отклика 0..3, прокси состояния)
    """
    state_proxy = (
        0.55 - dr + 0.0025 * fc + 0.18 * np.maximum(il, 0.0) - 0.010 * ip - 0.035 * np.log(np.maximum(ocr, 1.0))
    )
    idx = np.select(
        [(ip >= 12.0) & (fc >= 35.0), state_proxy > 0.14, state_proxy < -0.08],
        [np.full_like(state_proxy, 3), np.full_like(state_proxy, 0), np.full_like(state_proxy, 2)],
        default=np.full_like(state_proxy, 1),
    ).astype(int)
    return idx, state_proxy.astype(np.float32)


def _cycle_slope(dr, fc, ip, il, ocr, damping_percent, frequency, a_static, response_idx) -> np.ndarray:
    """
    Показатель циклической деградации s = 1 − α.

    :param dr: относительная плотность Dr
    :param fc: содержание мелких частиц, %
    :param ip: число пластичности Ip, %
    :param il: показатель текучести Il
    :param ocr: коэффициент переуплотнения
    :param damping_percent: демпфирование, %
    :param frequency: частота, Гц
    :param a_static: начальный статический сдвиг
    :param response_idx: индексы типа отклика
    :return: показатель s
    """
    plastic_blend = _sigmoid(ip, 8.0, 4.0) * _sigmoid(fc, 30.0, 12.0)
    slope = 0.305 - 0.075 * dr - 0.105 * plastic_blend
    slope = slope + np.select(
        [response_idx == 0, response_idx == 2, response_idx == 3],
        [0.035, -0.030, -0.060],
        default=0.0,
    )
    slope = slope + 0.018 * np.clip(il, 0.0, 1.5)
    slope = slope - 0.014 * np.log(np.maximum(ocr, 1.0))
    slope = slope + np.where(
        (a_static > 0.0) & (response_idx == 0), 0.045 * np.clip(a_static / 0.25, 0.0, 1.0), 0.0
    )
    slope = slope + 0.0030 * np.maximum(damping_percent - 5.0, 0.0)
    slope = slope - np.where(ip >= 8.0, 0.010 * np.log10(np.maximum(frequency, 0.02)), 0.0)
    return np.clip(slope, 0.08, 0.42)


def compute_crr_components(
    Dr: np.ndarray,
    Ip: np.ndarray,
    Il: np.ndarray,
    Ir: np.ndarray,
    fines_content: np.ndarray,
    clay_fraction: np.ndarray,
    sigma_eff: np.ndarray,
    OCR: np.ndarray,
    K0: np.ndarray,
    Vs1: np.ndarray,
    static_shear_ratio: np.ndarray,
    cementation_index: np.ndarray,
    aging_years: np.ndarray,
    saturation: np.ndarray,
    B: np.ndarray,
    damping_percent: np.ndarray,
    frequency: np.ndarray,
    loading_mode_factor: np.ndarray,
    reference_cycle: float = 15.0,
) -> Dict[str, np.ndarray]:
    """
    Вычислить параметры физической модели CRR для всей популяции грунтов.

    По физико-механическим свойствам строится базовый уровень CRR15 и набор поправочных
    множителей, из которых получаются CRR_ref, наклон s, а также параметры кривой
    CRR(N) = β / N^(1 − α): ``alpha`` и ``betta``. Дополнительно возвращаются тип отклика
    и разложение по факторам — это и есть «параметры, которые строят CRR».

    :param Dr: относительная плотность, доли единицы
    :param Ip: число пластичности, %
    :param Il: показатель текучести
    :param Ir: содержание органики, %
    :param fines_content: содержание мелких частиц (< 0.075 мм), %
    :param clay_fraction: содержание глинистых частиц (< 0.002 мм), %
    :param sigma_eff: эффективное напряжение, кПа
    :param OCR: коэффициент переуплотнения
    :param K0: коэффициент бокового давления покоя
    :param Vs1: скорректированная скорость поперечных волн, м/с
    :param static_shear_ratio: начальный статический сдвиг
    :param cementation_index: индекс цементации (0…3)
    :param aging_years: возраст грунта, лет
    :param saturation: степень водонасыщения, доли единицы
    :param B: параметр Скемптона B, доли единицы
    :param damping_percent: демпфирование, %
    :param frequency: частота нагружения, Гц
    :param loading_mode_factor: множитель режима лабораторного нагружения
    :param reference_cycle: опорное число циклов N_ref (обычно 15)
    :return: словарь массивов: ``alpha``, ``betta``, ``crr_ref``, ``cycle_slope``,
             ``response_idx``, ``state_proxy`` и факторы ``f_*``
    """
    Dr = np.clip(np.asarray(Dr, dtype=np.float64), 0.05, 1.0)
    Ip = np.maximum(np.asarray(Ip, dtype=np.float64), 0.0)
    Il = np.asarray(Il, dtype=np.float64)
    fc = np.clip(np.asarray(fines_content, dtype=np.float64), 0.0, 100.0)
    clay = np.clip(np.asarray(clay_fraction, dtype=np.float64), 0.0, 100.0)
    ocr = np.maximum(np.asarray(OCR, dtype=np.float64), 0.35)
    a_static = np.clip(np.asarray(static_shear_ratio, dtype=np.float64), 0.0, 0.45)

    response_idx, state_proxy = _infer_response(Dr, fc, Ip, Il, ocr)
    anisotropy = np.clip(1.0 - 0.035 * (np.asarray(K0, dtype=np.float64) - 0.5), 0.94, 1.05)

    factors = {
        "f_density_crr15": _base_crr15_from_density(Dr),
        "f_fines_plasticity": _fines_plasticity_factor(fc, Ip, clay),
        "f_liquidity": _liquidity_factor(Il),
        "f_organic": _organic_factor(np.asarray(Ir, dtype=np.float64)),
        "f_overburden": _overburden_factor(np.asarray(sigma_eff, dtype=np.float64), Dr),
        "f_ocr": _ocr_factor(ocr),
        "f_vs1": _vs_factor(np.asarray(Vs1, dtype=np.float64)),
        "f_cementation_aging": _cementation_aging_factor(
            np.asarray(cementation_index, dtype=np.float64), np.asarray(aging_years, dtype=np.float64)
        ),
        "f_static_shear": _static_shear_factor(a_static, response_idx, Dr),
        "f_loading_mode": np.asarray(loading_mode_factor, dtype=np.float64),
        "f_saturation": _saturation_factor(
            np.asarray(saturation, dtype=np.float64), np.asarray(B, dtype=np.float64)
        ),
        "f_damping": _damping_factor(np.asarray(damping_percent, dtype=np.float64)),
        "f_frequency": _frequency_factor(np.asarray(frequency, dtype=np.float64), Ip, fc),
        "f_k0_anisotropy": anisotropy,
    }

    crr_ref = factors["f_density_crr15"].copy()
    for key, value in factors.items():
        if key != "f_density_crr15":
            crr_ref = crr_ref * value
    crr_ref = np.clip(crr_ref, 0.030, 0.900)

    slope = _cycle_slope(
        Dr, fc, Ip, Il, ocr, np.asarray(damping_percent, dtype=np.float64),
        np.asarray(frequency, dtype=np.float64), a_static, response_idx,
    )
    ref_cycle = max(float(reference_cycle), 1.0)
    alpha = np.clip(1.0 - slope, 0.58, 0.94)
    betta = np.clip(crr_ref * np.power(ref_cycle, slope), 0.045, 1.40)

    result = {
        "alpha": alpha,
        "betta": betta,
        "crr_ref": crr_ref,
        "cycle_slope": slope,
        "response_idx": response_idx,
        "state_proxy": state_proxy,
        "reference_cycle": np.full_like(alpha, ref_cycle),
    }
    result.update(factors)
    return result


def crr_curve(cycles: np.ndarray, alpha: np.ndarray, betta: np.ndarray) -> np.ndarray:
    """
    Построить кривую CRR(N) = β / N^(1 − α) по параметрам α и β.

    :param cycles: сетка числа циклов, форма (n, seq_len) или (seq_len,)
    :param alpha: параметр α, форма (n,)
    :param betta: параметр β, форма (n,)
    :return: значения CRR(N) той же формы, что и ``cycles``
    """
    cycles = np.maximum(np.asarray(cycles, dtype=np.float64), 1.0)
    alpha = np.asarray(alpha, dtype=np.float64)
    betta = np.asarray(betta, dtype=np.float64)
    if cycles.ndim == 2:
        exponent = (1.0 - alpha)[:, None]
        return (betta[:, None] / np.power(cycles, exponent)).astype(np.float32)
    return (betta / np.power(cycles, 1.0 - alpha)).astype(np.float32)
