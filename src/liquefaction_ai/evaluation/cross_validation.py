"""
P0-a: кросс-валидация по ОБЪЕКТАМ (leakage-free) с доверительными интервалами.

Импортируемые функции для ноутбука 3_4_object_cv_and_ci. Два протокола:
  * primary — stratified grouped K-fold (make_grouped_cv_folds);
  * secondary — leave-one-object-out (make_loo_object_folds).

API:
    build_folds(meta, config, seed, loo=False, n_splits=5) -> list[fold-dict]
    evaluate_fold(pop, config, fold_split, fold_id, device, models=DEFAULT_MODELS,
                  quick=False, seed=42, nested=False) -> (rows_df, samples_df)
    aggregate_cv(raw_df, metric_keys=None) -> summary_df (mean ± 95% CI по фолдам)

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
                                                split_conformal_coverage,
                                                simultaneous_conformal_coverage,
                                                per_trajectory_nonconformity,
                                                conformal_band_quantile)
from liquefaction_ai.evaluation.p3_ranking import compute_p3_score
from liquefaction_ai.training.persistence import load_model_metadata
from liquefaction_ai.training.loop import train_model
from liquefaction_ai import models as M

# (артефакт, отображаемое имя, физическая ли модель, тип тренировки 'torch'|'fit')
# Включены САМЫЕ ОПАСНЫЕ конкуренты в PRIMARY CV Transformer, Neural Spline
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

# Сетки для NESTED object-CV: селекция гиперпараметров ВНУТРИ каждого outer-фолда (по его
# train/val), а не один раз глобально — устраняет selection leakage в outer-test. Компактные,
# чтобы вложенный перебор (фолды × модели × сетка) оставался вычислительно выполнимым.
NESTED_GRIDS = {
    # Структурные модели — значимые ручки ёмкости/архитектуры. ВАЖНО: calibration_steps ЗАФИКСИРОВАН
    # на 0 для PRIMARY CV. Внутренняя градиентная доводка θ (>0 шагов) НЕ входит в нормированную
    # плотность потока (её якобиан не учтён) → mixture-NLL/coverage были бы невалидны для headline.
    # Эффект 1/2 шагов измеряется ОТДЕЛЬНО как абляция (ablation_study: calib_steps_1/2), не в headline.
    "dpi_flow": {"hidden_dim": [128, 160, 192],
                 "use_traj_residual": [True, False]},
    "evt_ssm": {"hidden_dim": [96, 128, 160, 192], "integrator": ["heun", "euler"],
                 "use_crr_damage": [True, False]},
    "dpi_evt": {"hidden_dim": [128, 160, 192], "crr_mode": ["decoupled", "hybrid", "damage"],
                 "calibration_steps": [0]},
    # Baselines тоже подбираются ВНУТРИ фолда (иначе у них selection-leakage, а у proposed — нет).
    "transformer": {"hidden_dim": [64, 96, 128]},
    # NSF не имеет hidden_dim — ёмкость регулируется числом coupling-слоёв и бинов сплайна.
    "nsf": {"n_layers": [4, 6, 8], "n_bins": [8, 12]},
    "gru": {"hidden_dim": [64, 96, 128]},
    "tcn": {"hidden_dim": [64, 96, 128]},
    "pinn": {"hidden_dim": [64, 96, 128]},
    "catboost": {"iterations": [300, 600], "depth": [4, 6, 8], "learning_rate": [0.03, 0.05]},
}
# Селекция по POST-PREFIX прогнозу (continuation), а не по full-horizon RMSE — это соответствует
# prefix-conditioned forecasting (выбирать модель надо по тому, что она реально прогнозирует).
NESTED_SELECT_METRIC = "Traj_RMSE_continuation"

METRIC_KEYS = ["P3_Core", "N_liq_logMAE", "N_liq_logMAE_liq", "Traj_RMSE", "Traj_RMSE_continuation",
               "Traj_RMSE_continuation_balanced", "Traj_RMSE_continuation_worst", "Traj_RMSE_worst",
               "AUROC", "AUPRC", "Brier", "ECE", "Coverage_90", "Coverage_90_splitconf", "Coverage_90_simul",
               "Coverage_90_splitconf_width", "Coverage_90_simul_width",
               "Traj_RMSE_late", "Calibration_Error", "Traj_CRPS",
               "Onset_EarlyWarning_Rate", "Physics_Violation_Rate",
               "CRR_RMSE", "N_CRR_test", "N_CRR_objects",
               "CRR_Onset_Coherence_MAE", "N_liq_Coverage_90_liq", "N_liq_Interval_Width_90_liq",
               "Traj_RMSE_continuation_siteMacro", "N_liq_logMAE_siteMacro", "N_sites_test"]
# Метрики, передаваемые в P³: используем POST-PREFIX (continuation) траекторию как primary —
# это соответствует prefix-conditioned forecasting (p3_ranking сам предпочтёт continuation_balanced).
_P3_INPUT_KEYS = ["N_liq_logMAE", "Traj_RMSE", "Traj_RMSE_balanced", "Traj_RMSE_continuation",
                  "Traj_RMSE_continuation_balanced", "Traj_RMSE_continuation_worst",
                  "Brier", "AUPRC", "Physics_Violation_Rate"]

_SAMPLE_COLS = ["repeat", "fold", "model", "sidx", "object", "site_id", "liq_label", "n_liq_observed",
                "risk_label_observed", "nliq_censor_valid", "continuation_valid", "regime",
                "risk_prob_pred", "traj_rmse", "traj_rmse_continuation", "traj_sse_continuation",
                "continuation_points", "coverage90", "coverage90_hits",
                "nliq_log_err", "physics_violation", "interval_width", "onset_timing_bias_cyc",
                "mean_pred_std_continuation", "nonconf_max", "conf_q_val", "conf_band_width"]


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
        base = dict(hp["model_kwargs"])
        grid = NESTED_GRIDS.get(name) if nested else None
        if not grid:
            set_global_seed(seed + fold_id)
            return cls(**base).fit(train, val)
        from itertools import product
        from liquefaction_ai.training.losses import risk_observation_mask
        keys = list(grid)
        obs = risk_observation_mask(val)
        mask = (obs.detach().cpu().numpy() > 0.5) if obs is not None else np.ones(len(val["label"]), bool)
        y = val["label"].detach().cpu().numpy()[mask]
        best_score, best_model = float("inf"), None
        for values in product(*[grid[k] for k in keys]):
            params = {**base, **dict(zip(keys, values))}
            set_global_seed(seed + fold_id)
            candidate = cls(**params).fit(train, val)
            p = candidate.forward_batch(val)["risk_prob"].detach().cpu().numpy()[mask]
            score = float(np.mean((p - y) ** 2))
            if score < best_score:
                best_score, best_model = score, candidate
        return best_model
    hp, _ = load_model_metadata(models_dir, name)
    cls = getattr(M, hp["model_type"])
    model_kwargs = hp["model_kwargs"]
    grid = NESTED_GRIDS.get(name) if nested else None
    if grid:
        # Подбираем ручки, которые конструктор модели РЕАЛЬНО принимает (сверка с СИГНАТУРОЙ, а не с
        # сохранёнными model_kwargs — иначе параметры с дефолтом, напр. NSF n_bins, ошибочно
        # отбрасывались бы как «отсутствующие»). Защита от unexpected-kwarg сохраняется.
        import inspect
        _accepts = set(inspect.signature(cls.__init__).parameters)
        _dropped = [k for k in grid if k not in _accepts]
        grid = {k: v for k, v in grid.items() if k in _accepts}
        if _dropped: # НЕ молча: если ключ не подходит модели — это видно (а не тихий no-op)
            print(f" [nested] '{name}': ключи grid {_dropped} не принимаются конструктором — пропущены.")
        if not grid:
            print(f" [nested] '{name}': подходящих ручек нет → используются ГЛОБАЛЬНЫЕ гиперпараметры.")
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
            # FOLD-LOCAL variance-scaling калибровка σ на VAL текущего фолда (НЕ conformal — без
            # конечновыборочной гарантии; empirical object-held-out audit — ниже). Сбой НЕ глотаем
            # молча: иначе модель попадёт в таблицу как «calibrated», хотя калибровка не выполнилась.
            # strict=True → fatal; иначе — громкий WARN.
            if trainer == "torch":
                try:
                    fit_interval_scale(model, val, config, device)
                except Exception as e:
                    if strict:
                        raise RuntimeError(f"калибровка интервалов '{disp}' упала на fold {fold_id}: "
                                           f"{type(e).__name__}: {e}") from e
                    print(f" [WARN] калибровка '{disp}' на fold {fold_id} не выполнилась — "
                          f"Coverage может быть некорректным: {type(e).__name__}: {e}")
            test_out = collect_outputs(model, test, config, device)
            met, sdf = compute_metrics(disp, test_out, test, config)
            # conformal-скоры считаем на POST-PREFIX (continuation) маске, как и headline-метрики:
            # наблюдаемый префикс модель видит как контекст и тривиально его накрывает.
            def _cont_mask(split):
                if "continuation_mask" in split:
                    return split["continuation_mask"].cpu().numpy()
                m = split["mask"].cpu().numpy()
                if "prefix_mask" in split:
                    return m * (1.0 - np.minimum(split["prefix_mask"].cpu().numpy(), 1.0))
                return m
            _test_cm = _cont_mask(test)
            # split-conformal покрытие@90 (per-fold ДИАГНОСТИКА): квантиль калибруется на VAL-объектах
            # фолда (disjoint с test), применяется к test. ОГОВОРКА: val используется ещё и для
            # early-stopping/селекции, поэтому это НЕ чистый split-conformal и не формальная гарантия.
            # Pointwise marginal; width публикуется рядом.
            if trainer == "torch" and "traj_logvar" in test_out:
                try:
                    vo = collect_outputs(model, val, config, device)
                    if "traj_logvar" in vo:
                        _args = (vo["traj_mean"], np.sqrt(np.exp(vo["traj_logvar"])),
                                 val["r_obs"].cpu().numpy(), _cont_mask(val),
                                 test_out["traj_mean"], np.sqrt(np.exp(test_out["traj_logvar"])),
                                 test["r_obs"].cpu().numpy(), _test_cm)
                        cov_sc, w_sc = split_conformal_coverage(*_args, level=0.90)
                        cov_si, w_si = simultaneous_conformal_coverage(*_args, level=0.90)
                        met["Coverage_90_splitconf"] = cov_sc
                        met["Coverage_90_splitconf_width"] = w_sc # sharpness рядом с покрытием
                        met["Coverage_90_simul"] = cov_si
                        met["Coverage_90_simul_width"] = w_si
                except Exception as e:
                    # НЕ глотаем молча: иначе Coverage просто исчезает из фолда. strict→fatal.
                    if strict:
                        raise RuntimeError(f"split-conformal '{disp}' упал на fold {fold_id}: "
                                           f"{type(e).__name__}: {e}") from e
                    print(f" [WARN] split-conformal '{disp}' на fold {fold_id} пропущен: "
                          f"{type(e).__name__}: {e}")
            # EMPIRICAL site-held-out coverage: q калибруется на VAL-объектах (disjoint с test),
            # покрытие меряется на TEST-объектах (каждый объект — в test ровно одного фолда, моделью
            # его НЕ видевшей). Пишем в sdf пер-траекторный TEST-скор + калиброванный q фолда; агрегат
            # across folds (aggregate_object_conformal) даёт эмпирическое покрытие + object-bootstrap CI.
            # Это НЕ formal finite-sample гарантия (val участвует и в early-stop), а честная оценка.
            if trainer == "torch" and "traj_logvar" in test_out:
                _nc = per_trajectory_nonconformity(
                    test_out["traj_mean"], np.sqrt(np.exp(test_out["traj_logvar"])),
                    test["r_obs"].cpu().numpy(), _test_cm)
                _qcal = float("nan")
                try:
                    vo = collect_outputs(model, val, config, device)
                    if "traj_logvar" in vo:
                        _vnc = per_trajectory_nonconformity(
                            vo["traj_mean"], np.sqrt(np.exp(vo["traj_logvar"])),
                            val["r_obs"].cpu().numpy(), _cont_mask(val))
                        # калибруем q на ПЕР-ПЛОЩАДОЧНЫХ скорах val (макс по образцам site_id) — ТОЙ
                        # ЖЕ единице обмениваемости, что и сплит (site_id), и test-score в
                        # aggregate_object_conformal. Две скважины одной площадки = ОДИН кластер.
                        _gc = "site_id" if "site_id" in val["meta"].columns else "object"
                        _vobj = val["meta"][_gc].to_numpy() if _gc in val["meta"].columns else None
                        if _vobj is not None and np.isfinite(_vnc).any():
                            _fin = np.isfinite(_vnc)
                            _vobj_scores = (pd.Series(_vnc[_fin]).groupby(_vobj[_fin]).max().to_numpy())
                            _qcal = conformal_band_quantile(_vobj_scores, level=0.90)
                        else:
                            _qcal = conformal_band_quantile(_vnc, level=0.90)
                except Exception as e:
                    if strict:
                        raise RuntimeError(f"band-quantile '{disp}' fold {fold_id}: "
                                           f"{type(e).__name__}: {e}") from e
                sdf = sdf.copy(); sdf["nonconf_max"] = _nc; sdf["conf_q_val"] = _qcal
                if "mean_pred_std_continuation" in sdf.columns:
                    sdf["conf_band_width"] = 2.0 * _qcal * sdf["mean_pred_std_continuation"]
            per_model[disp] = met
            samples.append(_samples_frame(sdf, disp, fold_id, repeat))
        except Exception as e:
            # По умолчанию (strict=True) НЕ глотаем: молчаливый пропуск модели на фолде даёт ей
            # меньше n_folds и смещённое среднее (нечестное сравнение). Чинить причину, а не прятать.
            if strict:
                raise RuntimeError(f"модель '{disp}' упала на fold {fold_id}: {type(e).__name__}: {e}") from e
            print(f" [WARN] модель '{disp}' пропущена на fold {fold_id}: {type(e).__name__}: {e}")

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
    """Свести per-fold метрики в mean и SD между фолдами; inferential CI считаются site-bootstrap."""
    cols = [c for c in (metric_keys or METRIC_KEYS) if c in raw_df.columns]
    agg = raw_df.groupby("model")[cols].agg(["mean", "std", "count"])
    # n_folds = число строк-фолдов модели (а НЕ count ненулевого P3_Core: у CatBoost P3 может быть
    # NaN → раньше n_folds=0). Берём размер группы, который не зависит от наличия конкретной метрики.
    fold_counts = raw_df.groupby("model").size()
    rows = []
    for model in agg.index:
        rec = {"model": model, "n_folds": int(fold_counts.loc[model])}
        for c in cols:
            mean = float(agg.loc[model, (c, "mean")]); cnt = int(agg.loc[model, (c, "count")])
            std = float(agg.loc[model, (c, "std")]) if cnt > 1 else 0.0
            rec[f"{c}_mean"] = round(mean, 4)
            rec[f"{c}_fold_sd"] = round(std, 4)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("P3_Core_mean", ascending=False).reset_index(drop=True)


def aggregate_object_conformal(samples_df: pd.DataFrame, level: float = 0.90,
                               n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    EMPIRICAL site-held-out coverage@``level`` с object-bootstrap CI (честная оценка, НЕ гарантия).

    Для каждого (repeat, fold, object) берётся пер-ОБЪЕКТНЫЙ скор ``s = max`` пер-траекторных
    nonconformity по образцам объекта (вся площадка внутри полосы) и калиброванный на VAL квантиль
    ``q`` этого фолда (``conf_q_val``, disjoint с test). Объект «покрыт», если ``s ≤ q``. Покрытие =
    доля покрытых объектных точек; 95% CI — бутстрэп по КЛАСТЕРАМ-объектам (а не по образцам). Так
    оценивается, накроет ли полоса НОВЫЙ объект; q калибруется вне test → не тавтология. При 20
    объектах честнее заявлять именно empirical coverage + CI, чем formal finite-sample гарантию.

    :param samples_df: per-sample выходы (нужны model/object/repeat/fold/nonconf_max/conf_q_val)
    :param level: целевой уровень
    :return: таблица с empirical coverage, object-bootstrap CI и фактической средней шириной полосы
    """
    # Единица кластера = ПЛОЩАДКА (site_id), а не отдельная скважина (object): сплит держит площадку
    # целиком в одном фолде, поэтому две скважины одной площадки — ОДИН кластер, не два независимых.
    ccol = "site_id" if (samples_df is not None and "site_id" in samples_df.columns) else "object"
    cols = {"model", ccol, "nonconf_max", "conf_q_val"}
    if samples_df is None or not cols.issubset(samples_df.columns):
        return pd.DataFrame(columns=["model", "Coverage_emp", "Coverage_lo", "Coverage_hi",
                                     "mean_band_q", "mean_band_width", "n_objects", "n_object_points"])
    df = samples_df.copy()
    if "repeat" in df.columns:
        df = df[df["repeat"] == df["repeat"].min()].copy()
    df = df[np.isfinite(df["nonconf_max"].astype(float)) & np.isfinite(df["conf_q_val"].astype(float))]
    gkeys = [c for c in ("repeat", "fold", ccol) if c in df.columns]
    rows = []
    for model, g in df.groupby("model"):
        # кластерная точка = (фолд, site_id): скор всей площадки vs её фолд-калиброванный q
        agg_spec = {"s": ("nonconf_max", "max"), "q": ("conf_q_val", "first"),
                    "obj": (ccol, "first")}
        if "conf_band_width" in g.columns:
            agg_spec["width"] = ("conf_band_width", "mean")
        per_obj = g.groupby(gkeys).agg(**agg_spec).reset_index(drop=True)
        if per_obj.empty:
            continue
        covered = (per_obj["s"].to_numpy() <= per_obj["q"].to_numpy()).astype(float)
        objs = per_obj["obj"].to_numpy()
        cov = float(covered.mean())
        # object-cluster bootstrap CI (ресэмпл уникальных объектов с возвратом)
        rng = np.random.default_rng(seed)
        uniq = np.unique(objs)
        by_obj = {o: covered[objs == o] for o in uniq}
        boots = []
        for _ in range(int(n_boot)):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            vals = np.concatenate([by_obj[o] for o in pick])
            boots.append(vals.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"model": model, "Coverage_emp": round(cov, 4),
                     "Coverage_lo": round(float(lo), 4), "Coverage_hi": round(float(hi), 4),
                     "mean_band_q": round(float(per_obj["q"].mean()), 4),
                     "mean_band_width": (round(float(per_obj["width"].mean()), 4)
                                         if "width" in per_obj else float("nan")),
                     "n_objects": int(len(uniq)),
                     "n_object_points": int(len(per_obj))})
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
