"""
P1: абляции DPI-Flow (component-contribution) с метриками ПО 3 СОСТОЯНИЯМ опыта.

Импортируемые функции для ноутбука 3_6_ablations. Каждая абляция 1:1 отключает один заявленный
компонент вклада; обучение/оценка — на объектном (leakage-free) фолде (тот же протокол, что P0).

API:
    run_ablation_fold(pop, config, fold_split, fold_id, feat_names, device,
                      quick=False, seed=42, only=None) -> rows_df
    aggregate_ablations(raw_df) -> summary_df (mean ± 95% CI по фолдам)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from liquefaction_ai.config import set_global_seed
from liquefaction_ai.data.splits import prepare_benchmark_dataset
from liquefaction_ai.evaluation.metrics import (collect_outputs, compute_metrics,
                                                fit_interval_scale, stress_split)
from liquefaction_ai.models.dpi_flow import DPIFlow
from liquefaction_ai.training.persistence import load_model_metadata
from liquefaction_ai.training.loop import train_model

VS_FEATURES = ["V_s", "Vs1"]
GRAIN_FEATURES = ["D_r", "I_p", "fines_content", "clay_fraction", "log10_Cu"]
CRR_DERIVED_FEATURES = ["crr_alpha", "crr_betta", "crr_ref", "crr_cycle_slope", "crr_state_proxy"]

# (имя, kwargs-оверрайды DPIFlow, conformal?, обнулить признаки, stress-режим данных)
# stress: None | 'no_prefix' (занулить наблюдаемый префикс) | 'no_aux' (убрать derived g_obs/risk_proxy)
VARIANTS = [
    ("full", {}, True, [], None),
    # wo_varscale: без пост-hoc VARIANCE-SCALING калибровки σ (calib_log_scale=0). Раньше называлось
    # «wo_conformal» — это МИСЛЕЙБЛ: fit_interval_scale масштабирует σ по покрытию на VAL, это НЕ
    # split-conformal (настоящий conformal — headline site-held-out в 3_4/aggregate_object_conformal).
    ("wo_varscale", {}, False, [], None),
    ("gaussian_posterior", {"use_flow": False}, True, [], None),
    # wo_ode: убран ODE-слой; монотонная проекция (cummax) ОСТАЁТСЯ (use_monotone_clip=True — дефолт).
    # Это и есть «black-box декодер + cummax», поэтому отдельный blackbox_cummax дублировал бы wo_ode.
    ("wo_ode", {"use_analytical_layer": False}, True, [], None),
    # «изотонический» контроль: тот же black-box БЕЗ ODE и БЕЗ cummax.
    # Пара wo_ode (с cummax) vs blackbox_raw (без) показывает, что выигрыш НЕ сводится к cummax-постпроцессингу.
    ("blackbox_raw", {"use_analytical_layer": False, "use_monotone_clip": False}, True, [], None),
    ("wo_monotone", {"use_monotone_clip": False}, True, [], None),
    # ЭВРИСТИЧЕСКАЯ test-time калибровка θ по префиксу (calibration_steps>0). ВАЖНО: у нормирующего
    # потока это НЕ часть плотности — якобиан внутренних градиентных шагов не учтён, поэтому mixture-NLL
    # после такой доводки некорректен. Поэтому headline DPI-Flow идёт с calibration_steps=0 (честная
    # плотность), а 1/2 шага показаны ЗДЕСЬ как абляции (conformal=True: покрытие меряем conformal-полосой,
    # а не плотностью потока). Пара calib_steps_1/2 vs full отвечает: помогает ли доводка точечному прогнозу.
    ("calib_steps_1", {"calibration_steps": 1}, True, [], None),
    ("calib_steps_2", {"calibration_steps": 2}, True, [], None),
    ("wo_risk_softauc", {"use_discriminative_risk": False}, True, [], None),
    ("wo_censored_nliq", {"use_censored_nliq": False}, True, [], None),
    ("miss_vs", {}, True, VS_FEATURES, None),
    ("miss_grainsize", {}, True, GRAIN_FEATURES, None),
    ("no_crr_features", {}, True, CRR_DERIVED_FEATURES, None),
    # Stress-тесты прогноз без наблюдаемого PPR-префикса и без derived-aux.
    ("no_prefix", {}, True, [], "no_prefix"),
    ("prefix_only", {}, True, [], "no_static"),
    ("no_aux", {"use_observed_aux_loss": False}, True, [], "no_aux"),
]

METRIC_KEYS = ["N_liq_logMAE", "N_liq_logMAE_liq", "Traj_RMSE", "Traj_RMSE_continuation", "Traj_RMSE_late",
               "Traj_RMSE_liq", "Traj_RMSE_stab", "Traj_RMSE_nostab",
               "Traj_RMSE_balanced", "Traj_RMSE_worst",
               "AUPRC", "AUROC", "Brier", "ECE", "Coverage_90",
               "Calibration_Error", "Physics_Violation_Rate", "Traj_CRPS"]


def _zero_features(bench: dict, names: List[str], feat_names: List[str]) -> None:
    """Обнулить (=среднее после стандартизации) указанные статические признаки во всех выборках."""
    if not names:
        return
    idx = [feat_names.index(n) for n in names if n in feat_names]
    if not idx:
        return
    for key in ("train", "val", "test"):
        bench[key]["static"][:, idx] = 0.0


def run_ablation_fold(pop: dict, config, fold_split: dict, fold_id: int, feat_names: List[str],
                      device, quick: bool = False, seed: int = 42, only: Optional[str] = None,
                      models_dir: str = "models", tag: str = "grouped") -> pd.DataFrame:
    """Прогнать все (или один) варианты абляции на одном объектном фолде."""
    hp, _ = load_model_metadata(models_dir, "dpi_flow")
    base_kwargs = dict(hp["model_kwargs"])
    epochs = config.physics_epochs if quick else getattr(config, "publication_physics_epochs", 80)
    rows = []
    for name, overrides, use_conf, drop_feats, stress in VARIANTS:
        if only and name != only:
            continue
        # свежий dataset на каждый вариант (обнуление признаков не протекает между вариантами)
        bench = prepare_benchmark_dataset(pop, config, device, precomputed_split=fold_split)
        _zero_features(bench, drop_feats, feat_names)
        train, val, test = bench["train"], bench["val"], bench["test"]
        if stress == "no_prefix": # стресс: убрать наблюдаемый PPR-префикс из всех выборок
            train = stress_split(train, no_prefix=True); val = stress_split(val, no_prefix=True)
            test = stress_split(test, no_prefix=True)
        elif stress == "no_aux": # стресс: убрать derived auxiliary g_obs/risk_proxy
            train = stress_split(train, drop_derived_aux=True); val = stress_split(val, drop_derived_aux=True)
            test = stress_split(test, drop_derived_aux=True)
        elif stress == "no_static":
            train = stress_split(train, no_static=True); val = stress_split(val, no_static=True)
            test = stress_split(test, no_static=True)
        set_global_seed(seed + fold_id)
        model = DPIFlow(**{**base_kwargs, **overrides}).to(device)
        model, _ = train_model(model, train, val, epochs=epochs, model_name=f"abl:{name}(f{fold_id})",
                               config=config, device=device, verbose=False, scheduler="cosine")
        if use_conf:
            # НЕ глотаем ошибку молча: иначе абляция «с калибровкой» может незаметно остаться
            # без неё и сравнение wo_variance_scaling станет бессмысленным. (Это variance-scaling
            # σ-калибровка, НЕ conformal — настоящий conformal отдельный, см. metrics.)
            fit_interval_scale(model, val, config, device)
        elif hasattr(model, "calib_log_scale"):
            with torch.no_grad():
                model.calib_log_scale.zero_()
        met, _ = compute_metrics(f"abl:{name}", collect_outputs(model, test, config, device), test, config)
        row = {"tag": tag, "fold": fold_id, "ablation": name}
        row.update({k: met.get(k) for k in METRIC_KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_ablations(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRIC_KEYS if c in raw_df.columns]
    agg = raw_df.groupby("ablation")[cols].agg(["mean", "std", "count"])
    rows = []
    for abl in agg.index:
        rec = {"ablation": abl, "n_folds": int(agg.loc[abl, (cols[0], "count")])}
        for c in cols:
            mean = float(agg.loc[abl, (c, "mean")]); cnt = int(agg.loc[abl, (c, "count")])
            std = float(agg.loc[abl, (c, "std")]) if cnt > 1 else 0.0
            rec[f"{c}_mean"] = round(mean, 4)
            rec[f"{c}_fold_sd"] = round(std, 4)
        rows.append(rec)
    return pd.DataFrame(rows)
