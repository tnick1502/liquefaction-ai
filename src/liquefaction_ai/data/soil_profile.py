"""
Генерация полного геотехнического профиля грунта.

Для каждого сценария сэмплируется тип грунта по ГОСТ, его гранулометрический состав,
полный набор физических свойств (плотности, влажность, пределы Аттерберга, пористость,
органика, карбонатность) и механических характеристик, включая все входы физической
модели CRR (Dr, e/e_min/e_max, Ip, содержание мелких/глинистых частиц, эффективное
напряжение, OCR, K0, Vs1, начальный статический сдвиг, цементация, старение, водонасыщение,
демпфирование), а также общие механические параметры (модуль деформации E, сцепление c,
угол трения φ). По этим свойствам через :mod:`liquefaction_ai.physics.crr_physical`
вычисляются параметры кривой CRR(N) = β / N^(1 − α) и разложение по физическим факторам.

Результат — таблица с полным набором параметров, пригодная для глубокого анализа, и
параметры CRR, физически связанные с грунтом.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from liquefaction_ai.data.grainsize import (
    FRACTION_KEYS,
    TYPE_GROUND_NAMES_EN,
    define_type_ground,
    plaxis_classification,
    sample_grain_size,
)
from liquefaction_ai.physics.crr_physical import RESPONSE_TYPES, compute_crr_components

__all__ = ["SOIL_TYPE_KEYS", "TYPE_GROUND_TO_KEY", "sample_soil_profiles"]

# Машиночитаемые ключи типов грунта (соответствуют кодам ГОСТ 1…9)
SOIL_TYPE_KEYS = [
    "gravelly_sand", "coarse_sand", "medium_sand", "fine_sand", "silty_sand",
    "sandy_loam", "loam", "clay", "peat",
]
TYPE_GROUND_TO_KEY = {i + 1: key for i, key in enumerate(SOIL_TYPE_KEYS)}
"""Соответствие кода ГОСТ type_ground машинному ключу типа грунта."""

# Параметрические таблицы по типам грунта (индекс = type_ground − 1)
_E_CENTER = np.array([0.60, 0.62, 0.65, 0.68, 0.72, 0.70, 0.85, 1.00, 2.50])
_E_SPREAD = np.array([0.12, 0.12, 0.13, 0.14, 0.16, 0.18, 0.22, 0.30, 0.80])
_E_MIN = np.array([0.38, 0.40, 0.42, 0.45, 0.48, 0.45, 0.50, 0.55, 1.00])
_E_MAX = np.array([0.85, 0.88, 0.90, 0.95, 1.05, 1.10, 1.30, 1.60, 4.00])
_RS_CENTER = np.array([2.68, 2.67, 2.66, 2.66, 2.67, 2.69, 2.71, 2.74, 1.80])
_RS_SPREAD = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.30])
_IP_CENTER = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 4.0, 12.0, 25.0, 22.0])
_IP_SPREAD = np.array([0.0, 0.0, 0.0, 0.0, 0.4, 2.0, 3.0, 7.0, 8.0])
_IR_CENTER = np.array([0.5, 0.5, 0.5, 0.7, 1.0, 1.5, 2.5, 3.5, 60.0])
_IR_SPREAD = np.array([0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 1.2, 1.5, 12.0])
_PHI_CENTER = np.array([38.0, 37.0, 35.0, 33.0, 30.0, 26.0, 22.0, 16.0, 14.0])
_PHI_SPREAD = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
_C_CENTER = np.array([1.0, 1.0, 2.0, 3.0, 6.0, 12.0, 25.0, 45.0, 30.0])
_C_SPREAD = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 10.0])
_E_MOD_CENTER = np.array([45.0, 42.0, 38.0, 32.0, 24.0, 20.0, 16.0, 12.0, 3.0])
_E_MOD_SPREAD = np.array([8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 3.0, 1.5])
_OCR_CENTER = np.array([1.1, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 2.8, 1.5])
_OCR_SPREAD = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.5])
_CEMENT_CENTER = np.array([0.30, 0.30, 0.20, 0.15, 0.10, 0.10, 0.15, 0.20, 0.00])
_CEMENT_SPREAD = np.array([0.40, 0.40, 0.30, 0.25, 0.20, 0.20, 0.25, 0.30, 0.05])
_WP_CENTER = np.array([0.0, 0.0, 0.0, 0.0, 12.0, 15.0, 18.0, 22.0, 40.0])
_VS_BASE = np.array([180.0, 170.0, 160.0, 150.0, 140.0, 135.0, 140.0, 150.0, 90.0])
_DAMP_CENTER = np.array([3.0, 3.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 10.0])
_DAMP_SPREAD = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])


def _by_type(table: np.ndarray, type_idx: np.ndarray) -> np.ndarray:
    """
    Выбрать значения параметрической таблицы по индексам типа грунта.

    :param table: таблица значений длины 9 (индекс = type_ground − 1)
    :param type_idx: массив индексов типа грунта (0…8)
    :return: массив значений, форма как у ``type_idx``
    """
    return table[type_idx]


def sample_soil_profiles(n: int, rng: np.random.Generator, type_ground_probs: np.ndarray) -> Dict[str, object]:
    """
    Сэмплировать полный геотехнический профиль для популяции грунтов.

    Возвращает таблицу свойств грунта (физика + механика + гранулометрия + классификации)
    и параметры физической модели CRR. Гранулометрия сэмплируется по архетипам типа грунта,
    тип уточняется функцией :func:`define_type_ground`, класс PLAXIS и характерные диаметры —
    функцией :func:`plaxis_classification`. CRR-параметры (α, β, CRR_ref, наклон, тип отклика
    и факторы) вычисляются на референсном режиме нагружения, то есть отражают собственную
    сопротивляемость грунта.

    :param n: число грунтов
    :param rng: генератор случайных чисел numpy
    :param type_ground_probs: вероятности типов грунта (длина 9, сумма 1)
    :return: словарь с ключами ``soil_df`` (таблица свойств) и ``crr`` (словарь массивов CRR)
    """
    type_target = rng.choice(9, size=n, p=type_ground_probs) + 1
    type_idx = type_target - 1

    fractions = sample_grain_size(type_target, rng)  # (n, 11), сумма 100

    Ip = np.clip(rng.normal(_by_type(_IP_CENTER, type_idx), np.maximum(_by_type(_IP_SPREAD, type_idx), 1e-6)), 0.0, 60.0)
    Ir = np.clip(rng.normal(_by_type(_IR_CENTER, type_idx), _by_type(_IR_SPREAD, type_idx)), 0.0, 100.0)

    type_ground = define_type_ground(fractions, Ip, Ir)
    plaxis = plaxis_classification(fractions)
    d10 = plaxis["D10"].astype(np.float64)

    frac = {key: fractions[:, i] for i, key in enumerate(FRACTION_KEYS)}
    clay_fraction = frac["0000"]
    fines_content = frac["001"] + frac["0002"] + frac["0000"] + 0.585 * frac["005"]

    rs = np.clip(rng.normal(_by_type(_RS_CENTER, type_idx), _by_type(_RS_SPREAD, type_idx)), 1.3, 2.85)
    e_min = _by_type(_E_MIN, type_idx)
    e_max = _by_type(_E_MAX, type_idx)
    e = np.clip(rng.normal(_by_type(_E_CENTER, type_idx), _by_type(_E_SPREAD, type_idx)), e_min + 0.02, e_max - 0.02)
    Dr = np.clip((e_max - e) / np.maximum(e_max - e_min, 1e-9), 0.05, 1.0)

    saturation = rng.uniform(0.85, 1.0, size=n)
    W = e * saturation / rs * 100.0
    rd = rs / (1.0 + e)
    r = rd * (1.0 + W / 100.0)
    porosity = e / (1.0 + e) * 100.0

    Wp = np.where(Ip > 0.5, np.clip(rng.normal(_by_type(_WP_CENTER, type_idx), 3.0), 5.0, 60.0), 0.0)
    Wl = np.where(Ip > 0.5, Wp + Ip, 0.0)
    Il = np.where(Ip > 0.5, (W - Wp) / np.maximum(Ip, 1e-6), 0.0)
    Il = np.clip(Il, -0.5, 2.0)

    phi = np.clip(rng.normal(_by_type(_PHI_CENTER, type_idx), _by_type(_PHI_SPREAD, type_idx)), 8.0, 45.0)
    cohesion = np.clip(rng.normal(_by_type(_C_CENTER, type_idx), _by_type(_C_SPREAD, type_idx)), 0.0, 120.0)
    E_modulus = np.clip(rng.normal(_by_type(_E_MOD_CENTER, type_idx), _by_type(_E_MOD_SPREAD, type_idx)), 1.0, 80.0)
    OCR = np.clip(rng.normal(_by_type(_OCR_CENTER, type_idx), _by_type(_OCR_SPREAD, type_idx)), 1.0, 8.0)
    K0 = np.clip((1.0 - np.sin(np.deg2rad(phi))) * np.power(OCR, 0.4), 0.35, 1.05)
    Vs1 = np.clip(_by_type(_VS_BASE, type_idx) + 180.0 * Dr + rng.normal(0.0, 15.0, size=n), 80.0, 450.0)
    static_shear_ratio = np.clip(np.abs(rng.normal(0.05, 0.06, size=n)), 0.0, 0.35)
    cementation_index = np.clip(rng.normal(_by_type(_CEMENT_CENTER, type_idx), _by_type(_CEMENT_SPREAD, type_idx)), 0.0, 3.0)
    aging_years = np.power(10.0, rng.uniform(0.0, 4.0, size=n))
    B_value = rng.uniform(0.88, 0.98, size=n)
    damping_percent = np.clip(rng.normal(_by_type(_DAMP_CENTER, type_idx), _by_type(_DAMP_SPREAD, type_idx)), 0.5, 15.0)

    depth = np.clip(np.power(10.0, rng.uniform(0.0, 1.55, size=n)), 1.0, 40.0)
    gwl = rng.uniform(0.0, depth)
    sigma_v_total = r * 9.81 * depth
    pore_u = np.maximum(depth - gwl, 0.0) * 9.81
    sigma_v_eff = np.clip(sigma_v_total - pore_u, 10.0, 500.0)

    calcite = np.clip(np.abs(rng.normal(3.0, 5.0, size=n)), 0.0, 45.0)
    dolomite = np.clip(np.abs(rng.normal(1.0, 2.0, size=n)), 0.0, 25.0)
    insoluble_residue = np.clip(100.0 - calcite - dolomite, 0.0, 100.0)

    permeability = np.clip(np.power(d10, 2.0) / 100.0, 1e-9, 1e-2)

    # CRR-параметры на референсном режиме (loading_mode=dss=1.0, frequency=1 Гц)
    crr = compute_crr_components(
        Dr=Dr, Ip=Ip, Il=Il, Ir=Ir, fines_content=fines_content, clay_fraction=clay_fraction,
        sigma_eff=sigma_v_eff, OCR=OCR, K0=K0, Vs1=Vs1, static_shear_ratio=static_shear_ratio,
        cementation_index=cementation_index, aging_years=aging_years, saturation=saturation,
        B=B_value, damping_percent=damping_percent, frequency=np.ones(n), loading_mode_factor=np.ones(n),
    )
    response_type = np.array(RESPONSE_TYPES, dtype=object)[crr["response_idx"]]

    data: Dict[str, np.ndarray] = {
        # Идентификация и классификация
        "class_id": (type_ground - 1).astype(int),
        "type_ground": type_ground.astype(int),
        "soil_type": np.array([TYPE_GROUND_TO_KEY[int(t)] for t in type_ground], dtype=object),
        "soil_name_en": np.array([TYPE_GROUND_NAMES_EN[int(t)] for t in type_ground], dtype=object),
        "plaxis_class": plaxis["plaxis_class"],
        "response_type": response_type,
        # Физические параметры
        "rs": rs.astype(np.float32), "r": r.astype(np.float32), "rd": rd.astype(np.float32),
        "n_porosity": porosity.astype(np.float32), "e": e.astype(np.float32),
        "e_min": e_min.astype(np.float32), "e_max": e_max.astype(np.float32),
        "W": W.astype(np.float32), "Wl": Wl.astype(np.float32), "Wp": Wp.astype(np.float32),
        "I_p": Ip.astype(np.float32), "Il": Il.astype(np.float32), "Ir": Ir.astype(np.float32),
        "D10": plaxis["D10"], "D50": plaxis["D50"], "D60": plaxis["D60"], "Cu": plaxis["Cu"],
        "fines_content": fines_content.astype(np.float32), "clay_fraction": clay_fraction.astype(np.float32),
        "calcite": calcite.astype(np.float32), "dolomite": dolomite.astype(np.float32),
        "insoluble_residue": insoluble_residue.astype(np.float32),
        "depth": depth.astype(np.float32), "ground_water_depth": gwl.astype(np.float32),
        # Механические параметры (вход CRR-модели)
        "D_r": Dr.astype(np.float32), "V_s": Vs1.astype(np.float32), "Vs1": Vs1.astype(np.float32),
        "sigma_eff": sigma_v_eff.astype(np.float32), "OCR": OCR.astype(np.float32), "K0": K0.astype(np.float32),
        "static_shear_ratio": static_shear_ratio.astype(np.float32),
        "cementation_index": cementation_index.astype(np.float32), "aging_years": aging_years.astype(np.float32),
        "saturation": saturation.astype(np.float32), "B_value": B_value.astype(np.float32),
        "damping_ratio": damping_percent.astype(np.float32), "xi": (damping_percent / 100.0).astype(np.float32),
        "permeability": permeability.astype(np.float32),
        # Общие механические характеристики
        "phi": phi.astype(np.float32), "cohesion": cohesion.astype(np.float32), "E_modulus": E_modulus.astype(np.float32),
        # Параметры физической модели CRR
        "crr_alpha": crr["alpha"].astype(np.float32), "crr_betta": crr["betta"].astype(np.float32),
        "crr_ref": crr["crr_ref"].astype(np.float32), "crr_cycle_slope": crr["cycle_slope"].astype(np.float32),
        "crr_state_proxy": crr["state_proxy"].astype(np.float32),
    }
    # Гранулометрические фракции
    for i, key in enumerate(FRACTION_KEYS):
        data[f"gran_{key}"] = fractions[:, i].astype(np.float32)
    # Факторы CRR (разложение)
    for key, value in crr.items():
        if key.startswith("f_"):
            data[f"crr_{key}"] = value.astype(np.float32)

    return {"soil_df": pd.DataFrame(data), "crr": crr}
