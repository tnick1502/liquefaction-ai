"""
Адаптер реальных лабораторных данных к формату артефакта проекта.

Позволяет подставить массивы реальных циклических испытаний вместо синтетических, не меняя
ноутбуки анализа, обучения и оценки. Принимает только **наблюдаемые** в опыте величины:
свойства грунта, историю нагружения (число циклов и CSR), измеренную траекторию порового
давления PPR(N), длину валидного участка, бинарную метку разжижения и число циклов до
разжижения N_liq. Синтетические «латентные» поля (скрытое повреждение z, триггер g, истинная
CRR, непрерывный риск-скор) не используются и не требуются.

Признаки, строящие CRR (α, β), вычисляются из физических свойств грунта физической моделью
:func:`liquefaction_ai.physics.crr_physical.compute_crr_components` — это входные признаки, а
не супервизия, поэтому доступны и на реальных данных.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.constants import LOAD_NAMES, SOIL_NAMES
from liquefaction_ai.data.observed import derive_observed_targets
from liquefaction_ai.data.splits import make_benchmark_splits
from liquefaction_ai.data.synthetic import build_feature_matrices
from liquefaction_ai.physics.crr_physical import RESPONSE_TYPES, compute_crr_components

__all__ = ["compute_crr_features", "enrich_crr_breakdown", "build_observed_prefix", "build_population_from_experiments"]


def _col(df: pd.DataFrame, name: str, default) -> np.ndarray:
    """Вернуть колонку как массив или заполнить значением по умолчанию при отсутствии."""
    if name in df.columns:
        return df[name].to_numpy()
    return np.full(len(df), default, dtype=float)


def compute_crr_features(soil_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Вычислить параметры кривой CRR (α, β) из физических свойств грунта.

    Использует физическую модель CRR на референсном режиме нагружения. Недостающие
    свойства заполняются разумными значениями по умолчанию.

    :param soil_df: таблица свойств грунта (Dr, Ip, fines, clay, σ′, OCR, K0, Vs1 и т.д.)
    :return: словарь с массивами ``crr_alpha``, ``crr_betta``, ``crr_ref``
    """
    n = len(soil_df)
    crr = compute_crr_components(
        Dr=_col(soil_df, "D_r", 0.5),
        Ip=_col(soil_df, "I_p", 0.0),
        Il=_col(soil_df, "Il", 0.0),
        Ir=_col(soil_df, "Ir", 0.0),
        fines_content=_col(soil_df, "fines_content", 10.0),
        clay_fraction=_col(soil_df, "clay_fraction", 2.0),
        sigma_eff=_col(soil_df, "sigma_eff", 100.0),
        OCR=_col(soil_df, "OCR", 1.0),
        K0=_col(soil_df, "K0", 0.5),
        Vs1=_col(soil_df, "Vs1", _col(soil_df, "V_s", 180.0)),
        static_shear_ratio=_col(soil_df, "static_shear_ratio", 0.0),
        cementation_index=_col(soil_df, "cementation_index", 0.0),
        aging_years=_col(soil_df, "aging_years", 1.0),
        saturation=_col(soil_df, "saturation", 1.0),
        B=_col(soil_df, "B_value", 0.95),
        damping_percent=_col(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0),
        frequency=np.ones(n),
        loading_mode_factor=np.ones(n),
    )
    return {"crr_alpha": crr["alpha"].astype(np.float32),
            "crr_betta": crr["betta"].astype(np.float32),
            "crr_ref": crr["crr_ref"].astype(np.float32)}


def enrich_crr_breakdown(soil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить к таблице свойств полное разложение CRR из физической модели.

    Дописывает колонки ``crr_alpha``, ``crr_betta``, ``crr_ref``, ``crr_cycle_slope``,
    ``crr_state_proxy``, ``response_type`` и факторы ``crr_f_*``. Это делает ноутбук анализа
    параметров CRR (серия 1.3) работоспособным и на реальных данных.

    :param soil_df: таблица свойств грунта
    :return: та же таблица с добавленными CRR-колонками
    """
    n = len(soil_df)
    crr = compute_crr_components(
        Dr=_col(soil_df, "D_r", 0.5), Ip=_col(soil_df, "I_p", 0.0), Il=_col(soil_df, "Il", 0.0),
        Ir=_col(soil_df, "Ir", 0.0), fines_content=_col(soil_df, "fines_content", 10.0),
        clay_fraction=_col(soil_df, "clay_fraction", 2.0), sigma_eff=_col(soil_df, "sigma_eff", 100.0),
        OCR=_col(soil_df, "OCR", 1.0), K0=_col(soil_df, "K0", 0.5),
        Vs1=_col(soil_df, "Vs1", _col(soil_df, "V_s", 180.0)),
        static_shear_ratio=_col(soil_df, "static_shear_ratio", 0.0),
        cementation_index=_col(soil_df, "cementation_index", 0.0), aging_years=_col(soil_df, "aging_years", 1.0),
        saturation=_col(soil_df, "saturation", 1.0), B=_col(soil_df, "B_value", 0.95),
        damping_percent=_col(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0),
        frequency=np.ones(n), loading_mode_factor=np.ones(n),
    )
    soil_df["crr_alpha"] = crr["alpha"].astype(np.float32)
    soil_df["crr_betta"] = crr["betta"].astype(np.float32)
    soil_df["crr_ref"] = crr["crr_ref"].astype(np.float32)
    soil_df["crr_cycle_slope"] = crr["cycle_slope"].astype(np.float32)
    soil_df["crr_state_proxy"] = crr["state_proxy"].astype(np.float32)
    soil_df["response_type"] = np.array(RESPONSE_TYPES, dtype=object)[crr["response_idx"]]
    for key, value in crr.items():
        if key.startswith("f_"):
            soil_df[f"crr_{key}"] = value.astype(np.float32)
    return soil_df


def build_observed_prefix(r_obs: np.ndarray, valid_mask: np.ndarray, prefix_len: int):
    """
    Сформировать наблюдаемый префикс из измеренной траектории (без добавления шума).

    :param r_obs: измеренная траектория PPR, форма (n, seq_len)
    :param valid_mask: маска валидной длины, форма (n, seq_len)
    :param prefix_len: длина префикса
    :return: словарь с ``prefix_obs`` и ``prefix_mask``
    """
    n, seq_len = r_obs.shape
    prefix_mask = ((np.arange(seq_len)[None, :] < prefix_len) & (valid_mask > 0)).astype(np.float32)
    prefix_obs = (r_obs * prefix_mask).astype(np.float32)
    return {"prefix_obs": prefix_obs, "prefix_mask": prefix_mask}


def build_population_from_experiments(
    soil_df: pd.DataFrame,
    load_df: pd.DataFrame,
    cycles: np.ndarray,
    csr: np.ndarray,
    r_measured: np.ndarray,
    valid_mask: np.ndarray,
    liq_label: np.ndarray,
    n_liq: np.ndarray,
    config: ExperimentConfig,
    crr_obs: Optional[np.ndarray] = None,
    crr_obs_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Собрать артефакт популяции из наблюдаемых данных реальных испытаний.

    Результат полностью совместим с :func:`liquefaction_ai.data.io.save_population_artifact` и
    :func:`liquefaction_ai.data.splits.prepare_benchmark_dataset`: содержит признаки, метаданные,
    измеренную траекторию и стратифицированное benchmark-разбиение. Синтетические латентные
    поля не создаются.

    Требования к входным таблицам:
    - ``soil_df`` — свойства грунта; обязательны колонки ``class_id`` (или ``type_ground``),
      ``soil_type`` и физ/мех колонки, используемые в признаках (``e, D_r, I_p, V_s, xi,
      sigma_eff, permeability, fines_content, clay_fraction, Cu``). Если ``crr_alpha``/``crr_betta``
      отсутствуют — вычисляются из свойств.
    - ``load_df`` — параметры нагружения: ``CSR_base, frequency, amp_scale, N_max,
      nonstationarity, load_mode, mode_id``.

    :param soil_df: таблица свойств грунта по испытаниям
    :param load_df: таблица параметров нагружения по испытаниям
    :param cycles: сетка числа циклов измерений, форма (n, seq_len)
    :param csr: история CSR на каждом узле, форма (n, seq_len)
    :param r_measured: измеренная траектория PPR, форма (n, seq_len)
    :param valid_mask: маска валидной длины измерений, форма (n, seq_len)
    :param liq_label: бинарная метка разжижения, форма (n,)
    :param n_liq: число циклов до разжижения (или N_max, если не разжижился), форма (n,)
    :param config: конфигурация эксперимента (длины, нормировки, размеры выборок)
    :param crr_obs: опциональная измеренная кривая CRR(N), форма (n, seq_len) (например, по серии
        из 6 образцов); используется как наблюдаемая супервизия границы CRR там, где доступна
    :param crr_obs_mask: опциональная по-образцовая маска наличия измеренной CRR, форма (n,)
    :return: словарь популяции в формате артефакта (только наблюдаемые поля)
    """
    soil_df = soil_df.reset_index(drop=True).copy()
    load_df = load_df.reset_index(drop=True).copy()
    n, seq_len = r_measured.shape

    # Производные классификационные колонки
    if "class_id" not in soil_df.columns and "type_ground" in soil_df.columns:
        soil_df["class_id"] = soil_df["type_ground"].astype(int) - 1
    if "soil_type" not in soil_df.columns:
        soil_df["soil_type"] = [SOIL_NAMES[int(i)] for i in soil_df["class_id"]]
    if "mode_id" not in load_df.columns and "load_mode" in load_df.columns:
        load_df["mode_id"] = load_df["load_mode"].map({name: i for i, name in enumerate(LOAD_NAMES)})

    # Полное разложение CRR из физической модели (входные признаки + диагностика)
    if "crr_alpha" not in soil_df.columns or "crr_betta" not in soil_df.columns:
        soil_df = enrich_crr_breakdown(soil_df)
    if "Cu" not in soil_df.columns:
        soil_df["Cu"] = 5.0
    if "Vs1" not in soil_df.columns:
        soil_df["Vs1"] = _col(soil_df, "V_s", 180.0)

    delta_cycles = np.diff(np.concatenate([np.zeros((n, 1)), cycles], axis=1), axis=1).astype(np.float32)
    observations = build_observed_prefix(r_measured.astype(np.float32), valid_mask.astype(np.float32), config.prefix_len)
    features = build_feature_matrices(soil_df, load_df, cycles.astype(np.float32), delta_cycles,
                                      csr.astype(np.float32), observations, config.prefix_len)

    meta = pd.concat([soil_df, load_df], axis=1)
    meta["liq_label"] = np.asarray(liq_label).astype(int)
    meta["N_liq_true"] = np.asarray(n_liq).astype(np.float32)
    meta["PPR_max_true"] = r_measured.max(axis=1)
    meta["CSR_max"] = csr.max(axis=1)

    # Наблюдаемые вспомогательные цели из измеренной кривой PPR
    obs_targets = derive_observed_targets(r_measured.astype(np.float32), valid_mask.astype(np.float32))

    benchmark = make_benchmark_splits(meta, min(config.benchmark_subset, n), config.seed, config)

    population = {
        "meta": meta,
        "cycles": cycles.astype(np.float32),
        "delta_cycles": delta_cycles,
        "csr": csr.astype(np.float32),
        "r_obs": r_measured.astype(np.float32),
        "valid_mask": valid_mask.astype(np.float32),
        "prefix_mask": observations["prefix_mask"],
        "prefix_obs": observations["prefix_obs"],
        "liq_label": np.asarray(liq_label).astype(np.float32),
        "n_liq_true": np.asarray(n_liq).astype(np.float32),
        "g_obs": obs_targets["g_obs"],
        "risk_proxy": obs_targets["risk_proxy"],
        "static_features": features["static_features"],
        "static_feature_names": features["static_feature_names"],
        "prefix_summary": features["prefix_summary"],
        "prefix_summary_names": features["prefix_summary_names"],
        "seq_inputs": features["seq_inputs"],
        "seq_feature_names": features["seq_feature_names"],
        "benchmark": benchmark,
    }
    # Опциональная измеренная кривая CRR(N) (если есть серия из 6 образцов)
    if crr_obs is not None:
        population["crr_obs"] = np.asarray(crr_obs, dtype=np.float32)
        population["crr_obs_mask"] = (np.ones(n, dtype=np.float32) if crr_obs_mask is None
                                      else np.asarray(crr_obs_mask, dtype=np.float32))
    return population
