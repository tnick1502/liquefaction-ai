"""
P0-a: кросс-валидация по ОБЪЕКТАМ (leakage-free) с доверительными интервалами.

Импортируемые функции для ноутбука 3_4_object_cv_and_ci. Два протокола:
  * primary   — stratified grouped K-fold (make_grouped_cv_folds);
  * secondary — leave-one-object-out (make_loo_object_folds).

API:
    build_folds(meta, config, seed, loo=False, n_splits=5) -> list[fold-dict]
    evaluate_fold(pop, config, fold_split, fold_id, device, models=DEFAULT_MODELS,
                  quick=False, seed=42, nested=False) -> (rows_df, samples_df)
    aggregate_cv(raw_df, metric_keys=None) -> summary_df  (mean ± 95% CI по фолдам)

При ``nested=True`` гиперпараметры структурных моделей подбираются ВНУТРИ каждого outer-фолда
(grid-search по его train/val, см. :data:`NESTED_GRIDS`), а не берутся глобально-фиксированными —
это делает CV истинно вложенным и снимает selection-leakage в outer-test. CI по фолдам/объектам
считаются как и раньше (``aggregate_cv`` + ``significance.object_cluster_bootstrap``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from liquefaction_ai.config import set_global_seed
from liquefaction_ai.data.splits import (make_grouped_cv_folds, make_loo_object_folds,
                                         prepare_benchmark_dataset)
from liquefaction_ai.evaluation.metrics import (collect_outputs, compute_metrics, fit_interval_scale,
                                                split_conformal_coverage)
from liquefaction_ai.evaluation.p3_ranking import compute_p3_score
from liquefaction_ai.training.persistence import load_model_metadata
from liquefaction_ai.training.loop import train_model
from liquefaction_ai import models as M

# (артефакт, отображаемое имя, физическая ли модель, тип тренировки 'torch'|'fit')
# Включены САМЫЕ ОПАСНЫЕ конкуренты в PRIMARY CV (замечание рецензента): Transformer, Neural Spline
# Flow (лучший raw RMSE), CatBoost (лучший Brier/N_liq), GRU/TCN — чтобы сильные baselines
# оценивались на том же объектном протоколе, а не только на слабом single-split.
DEFAULT_MODELS = [
    ("dpi_flow", "DPI-Flow", True, "torch"),
    ("dpi_evt", "DPI-EVT", True, "torch"),
    ("evt_ssm", "EVT-NeuralSSM", True, "torch"),
    ("pinn", "PINN", False, "torch"),
    ("transformer", "Transformer", False, "torch"),
    ("nsf", "Neural Spline Flow", False, "torch"),
    ("gru", "GRU", False, "torch"),
    ("tcn", "TCN", False, "torch"),
    ("catboost", "CatBoost", False, "fit"),
]
# Опорная (reference) модель для P³-нормировки.
P3_REFERENCE = "PINN"

# Сетки для NESTED object-CV (#2): селекция гиперпараметров ВНУТРИ каждого outer-фолда (по его
# train/val), а не один раз глобально — устраняет selection leakage в outer-test. Компактные,
# чтобы вложенный перебор (фолды × модели × сетка) оставался вычислительно выполнимым.
NESTED_GRIDS = {
    # Структурные модели — все значимые ручки (ёмкость, калибровка θ, residual, схема ODE, CRR).
    "dpi_flow": {"hidden_dim": [128, 160, 192], "calibration_steps": [1, 2],
                 "calibration_lr": [0.05, 0.10], "use_traj_residual": [True, False]},
    "evt_ssm":  {"hidden_dim": [96, 128, 160, 192], "integrator": ["heun", "euler"],
                 "use_crr_damage": [True, False]},
    "dpi_evt":  {"hidden_dim": [128, 160, 192], "crr_mode": ["decoupled", "hybrid", "damage"],
                 "calibration_steps": [0, 1]},
    # Baselines тоже подбираются ВНУТРИ фолда (иначе у них selection-leakage, а у proposed — нет).
    "transformer": {"hidden_dim": [64, 96, 128]},
    "nsf":         {"hidden_dim": [64, 96, 128]},
    "gru":         {"hidden_dim": [64, 96, 128]},
    "tcn":         {"hidden_dim": [64, 96, 128]},
    "pinn":        {"hidden_dim": [64, 96, 128]},
}
# Селекция по POST-PREFIX прогнозу (continuation), а не по full-horizon RMSE — это соответствует
# prefix-conditioned forecasting (выбирать модель надо по тому, что она реально прогнозирует).
NESTED_SELECT_METRIC = "Traj_RMSE_continuation"

METRIC_KEYS = ["P3_Core", "N_liq_logMAE", "N_liq_logMAE_liq", "Traj_RMSE", "Traj_RMSE_continuation",
               "Traj_RMSE_continuation_balanced", "Traj_RMSE_continuation_worst", "Traj_RMSE_worst",
               "AUROC", "AUPRC", "Brier", "ECE", "Coverage_90", "Coverage_90_splitconf",
               "Traj_RMSE_late", "Calibration_Error", "Traj_CRPS",
               "Onset_EarlyWarning_Rate", "Physics_Violation_Rate",
               "CRR_RMSE", "N_CRR_test", "N_CRR_objects"]   # CRR-метрики в primary CV (замечание рецензента)

# Метрики, передаваемые в P³: используем POST-PREFIX (continuation) траекторию как primary —
# это соответствует prefix-conditioned forecasting (p3_ranking сам предпочтёт continuation_balanced).
_P3_INPUT_KEYS = ["N_liq_logMAE", "Traj_RMSE", "Traj_RMSE_balanced", "Traj_RMSE_continuation",
                  "Traj_RMSE_continuation_balanced", "Traj_RMSE_continuation_worst",
                  "Brier", "AUPRC", "Physics_Violation_Rate"]

_SAMPLE_COLS = ["repeat", "fold", "model", "sidx", "object", "liq_label", "n_liq_observed",
                "risk_prob_pred", "traj_rmse", "traj_rmse_continuation", "coverage90",
                "nliq_log_err", "physics_violation", "interval_width", "onset_timing_bias_cyc"]


def build_folds(meta: pd.DataFrame, config, seed: int = 42, loo: bool = False,
                n_splits: int = 5, n_repeats: int = 1):
    n = min(config.benchmark_subset, len(meta))
    if loo:
        return make_loo_object_folds(meta, n, seed, config)
    return make_grouped_cv_folds(meta, n, seed, config, n_splits=n_splits, n_repeats=n_repeats)


def _samples_frame(sdf: pd.DataFrame, model: str, fold: int, repeat: int) -> pd.DataFrame:
    df = sdf.copy()
    df["repeat"] = repeat; df["fold"] = fold; df["model"] = model; df["sidx"] = np.arange(len(df))
    return df[[c for c in _SAMPLE_COLS if c in df.columns]]


def _train_one(name: str, disp: str, is_phys: bool, trainer: str, train, val,
               config, device, fold_id: int, seed: int, models_dir: str, nested: bool = False):
    """Обучить одну модель (torch-цикл или нативный .fit для табличных вроде CatBoost).

    При ``nested=True`` для моделей из :data:`NESTED_GRIDS` гиперпараметры подбираются ВНУТРИ
    фолда коротким grid-search по его train/val (а не берутся глобально-фиксированные) — это
    делает CV вложенным и снимает selection leakage в outer-test.
    """
    if trainer == "fit":
        # Не-torch (CatBoost): у него нет history.parquet, поэтому читаем hyperparams напрямую,
        # конструируем и обучаем нативным .fit (без train_model / .to(device)).
        hp = json.loads((Path(models_dir) / name / "hyperparams.json").read_text(encoding="utf-8"))
        cls = getattr(M, hp["model_type"])
        set_global_seed(seed + fold_id)
        model = cls(**hp["model_kwargs"])
        model.fit(train, val)
        return model
    hp, _ = load_model_metadata(models_dir, name)
    cls = getattr(M, hp["model_type"])
    model_kwargs = hp["model_kwargs"]
    grid = NESTED_GRIDS.get(name) if nested else None
    if grid:
        # Подбираем ТОЛЬКО те ручки, что реально есть в model_kwargs модели (защита от unexpected-kwarg).
        _dropped = [k for k in grid if k not in hp["model_kwargs"]]
        grid = {k: v for k, v in grid.items() if k in hp["model_kwargs"]}
        if _dropped:   # НЕ молча: если ключ не подходит модели — это видно (а не тихий no-op)
            print(f"  [nested] '{name}': ключи grid {_dropped} отсутствуют в model_kwargs — пропущены.")
        if not grid:
            print(f"  [nested] '{name}': подходящих ручек нет → используются ГЛОБАЛЬНЫЕ гиперпараметры.")
    if grid:
        # Внутренняя селекция: короткий grid-search по train/val ТЕКУЩЕГО фолда (outer-test не виден).
        from liquefaction_ai.training.search import grid_search
        base = {k: v for k, v in hp["model_kwargs"].items() if k not in grid}
        _, best = grid_search(lambda p: cls(**{**base, **p}), grid, train, val, config, device,
                              search_epochs=getattr(config, "grid_search_epochs", 8),
                              score_metric=NESTED_SELECT_METRIC)
        model_kwargs = {**base, **best}
    set_global_seed(seed + fold_id)
    model = cls(**model_kwargs).to(device)
    epochs = (getattr(config, "publication_physics_epochs", 200) if is_phys
              else getattr(config, "publication_baseline_epochs", 120))
    # Серьёзное обучение: косинусный LR с прогревом ДЛЯ ВСЕХ torch-моделей + ранняя остановка
    # по best-val (patience/min_delta из config). Потолок эпох высокий — реально остановит ES.
    model, _ = train_model(model, train, val, epochs=epochs, model_name=f"{disp}(f{fold_id})",
                           config=config, device=device, verbose=False, scheduler="cosine")
    return model


def evaluate_fold(pop: Dict, config, fold_split: Dict, fold_id: int, device,
                  models: List = None, quick: bool = False, seed: int = 42,
                  models_dir: str = "models", nested: bool = False,
                  strict: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Обучить модели на одном фолде и вернуть (per-model метрики, per-sample выходы).

    Возвращаемые таблицы содержат поля ``repeat``/``fold`` для корректной агрегации repeated CV и
    object-cluster bootstrap (см. evaluation.significance.object_cluster_bootstrap).
    """
    models = models or DEFAULT_MODELS
    repeat = int(fold_split.get("repeat", 0))
    bench = prepare_benchmark_dataset(pop, config, device, precomputed_split=fold_split)
    train, val, test = bench["train"], bench["val"], bench["test"]
    per_model, samples = {}, []
    for name, disp, is_phys, trainer in models:
        try:
            if quick and trainer == "torch":
                # дымовой режим: короткие demo-эпохи
                hp, _ = load_model_metadata(models_dir, name)
                cls = getattr(M, hp["model_type"]); set_global_seed(seed + fold_id)
                model = cls(**hp["model_kwargs"]).to(device)
                ep = config.physics_epochs if is_phys else config.baseline_epochs
                model, _ = train_model(model, train, val, epochs=ep, model_name=f"{disp}(f{fold_id})",
                                       config=config, device=device, verbose=False,
                                       scheduler="cosine" if is_phys else "none")
            else:
                model = _train_one(name, disp, is_phys, trainer, train, val, config, device,
                                   fold_id, seed, models_dir, nested=nested)
            # FOLD-LOCAL конформная калибровка интервалов на VAL текущего outer-фолда (замечание
            # рецензента: без неё Coverage@90 систематически недопокрывает → claim «calibrated» провален).
            if trainer == "torch":
                try:
                    fit_interval_scale(model, val, config, device)
                except Exception:
                    pass
            test_out = collect_outputs(model, test, config, device)
            met, sdf = compute_metrics(disp, test_out, test, config)
            # #9 DEPLOYABLE split-conformal покрытие@90: конформный квантиль калибруется на VAL-объектах
            # фолда (disjoint и с train, и с test), применяется к test → честное object-held-out покрытие
            # (не transductive). Pointwise marginal. Считается для torch-моделей с предсказанным σ.
            if trainer == "torch" and "traj_logvar" in test_out:
                try:
                    vo = collect_outputs(model, val, config, device)
                    if "traj_logvar" in vo:
                        cov_sc, _ = split_conformal_coverage(
                            vo["traj_mean"], np.sqrt(np.exp(vo["traj_logvar"])),
                            val["r_obs"].cpu().numpy(), val["mask"].cpu().numpy(),
                            test_out["traj_mean"], np.sqrt(np.exp(test_out["traj_logvar"])),
                            test["r_obs"].cpu().numpy(), test["mask"].cpu().numpy(), level=0.90)
                        met["Coverage_90_splitconf"] = cov_sc
                except Exception:
                    pass
            per_model[disp] = met
            samples.append(_samples_frame(sdf, disp, fold_id, repeat))
        except Exception as e:
            # По умолчанию (strict=True) НЕ глотаем: молчаливый пропуск модели на фолде даёт ей
            # меньше n_folds и смещённое среднее (нечестное сравнение). Чинить причину, а не прятать.
            if strict:
                raise RuntimeError(f"модель '{disp}' упала на fold {fold_id}: {type(e).__name__}: {e}") from e
            print(f"  [WARN] модель '{disp}' пропущена на fold {fold_id}: {type(e).__name__}: {e}")

    df_p3 = pd.DataFrame([{"model": d, **{k: per_model[d].get(k) for k in _P3_INPUT_KEYS}}
                          for d in per_model])
    try:
        ref = P3_REFERENCE if P3_REFERENCE in df_p3["model"].values else df_p3["model"].iloc[0]
        scored = compute_p3_score(df_p3, ref, "core")
        p3 = dict(zip(scored["model"], scored["P3_Core_Raw_Score"]))
    except Exception:
        p3 = {d: float("nan") for d in per_model}

    rows = []
    for disp, met in per_model.items():
        row = {"repeat": repeat, "fold": fold_id,
               "test_object": str(fold_split.get("test_object", "")), "model": disp,
               "P3_Core": p3.get(disp, float("nan"))}
        row.update({k: met.get(k) for k in METRIC_KEYS if k != "P3_Core"})
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True)


def aggregate_cv(raw_df: pd.DataFrame, metric_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """Свести per-fold метрики в mean ± 95% CI по фолдам."""
    cols = [c for c in (metric_keys or METRIC_KEYS) if c in raw_df.columns]
    agg = raw_df.groupby("model")[cols].agg(["mean", "std", "count"])
    rows = []
    for model in agg.index:
        rec = {"model": model, "n_folds": int(agg.loc[model, (cols[0], "count")])}
        for c in cols:
            mean = float(agg.loc[model, (c, "mean")]); cnt = int(agg.loc[model, (c, "count")])
            std = float(agg.loc[model, (c, "std")]) if cnt > 1 else 0.0
            rec[f"{c}_mean"] = round(mean, 4)
            rec[f"{c}_ci95"] = round(1.96 * std / np.sqrt(max(cnt, 1)), 4)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("P3_Core_mean", ascending=False).reset_index(drop=True)
