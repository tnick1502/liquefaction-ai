"""
P³-Score и P³-Pareto Ranking — публикационное ранжирование моделей разжижения грунта.

P³-Score = **Predictive–Probabilistic–Physical Score** (интегральная предсказательно-вероятностно-
физическая метрика качества). Работает поверх уже посчитанной таблицы метрик (``metrics_df`` из
:func:`liquefaction_ai.evaluation.compute_metrics`) и НЕ изменяет существующие функции.

Принципы (уровень топ-конференции):
- В score входит только небольшой **непересекающийся** набор метрик. Дублирующие/диагностические
  метрики (AUROC, ECE, Coverage_*, Interval_Width_*, Traj_NLL, raw N_liq_MAE/RMSE, Traj_MSE, val_loss)
  остаются в отчётной таблице, но НЕ в score.
- Нормировка **относительно фиксированной reference-модели** (100 = уровень reference) — это
  взвешенное геометрическое среднее относительных улучшений (shifted-geometric-mean profile).
- **AUPRC** имеет малый вес (≈0.10): при насыщении ~0.99–1.00 он не различает модели и не должен
  поднимать физически плохие модели наверх.
- **Физическая допустимость** — строгий gate: модель, выдавшая хотя бы одну физически невозможную
  кривую (Physics_Violation_Rate > 0), исключается из admissible ranking. Для
  структурных моделей нулевые нарушения — архитектурная гарантия монотонной feasible-кривой, а не
  отдельное доказательство физической истинности.
- Траекторная ось по возможности использует **post-prefix continuation RMSE**: задача проекта
  prefix-conditioned, поэтому reconstruction-RMSE по всему горизонту остаётся диагностикой.
- Главный результат — **admissible Pareto-фронт** + ``P3_*_Admissible_Score``; raw-версии —
  диагностические.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from liquefaction_ai.evaluation.metrics import METRICS, MetricInfo, METRIC_COLUMN_EN

__all__ = [
    "metric_direction", "compute_physical_admissibility", "compute_p3_score",
    "build_pareto_objectives", "pareto_rank", "publication_ranking_table",
]

# Веса непересекающихся критериев P³ по режимам (направление берётся из metric_direction).
# Траекторный компонент core берётся как BALANCED по трём состояниям опыта (разжижение /
# нет+стабилизация / нет+нет стабилизации), а не как глобальный pooled-RMSE: иначе модель может
# «спрятать» провал целого режима за счёт лёгкого большинства. Запасной вариант (нет колонки) —
# глобальный Traj_RMSE (см. _resolve_weights).
_P3_WEIGHTS: Dict[str, Dict[str, float]] = {
    "core": {"N_liq_logMAE": 0.45, "Traj_RMSE_balanced": 0.30, "Brier": 0.20, "AUPRC": 0.05},
    "probabilistic": {"N_liq_logMAE": 0.25, "Traj_CRPS": 0.25, "Brier": 0.20,
                      "Calibration_Error": 0.20, "AUPRC": 0.10},
    "physics": {"N_liq_logMAE": 0.25, "__TRAJ__": 0.25, "CRR_RMSE": 0.20, "Brier": 0.15,
                "Calibration_Error": 0.15},
}
# Объективы Pareto (все приводятся к «меньше — лучше»; AUPRC → 1 − AUPRC).
_P3_PARETO: Dict[str, List[str]] = {
    "core": ["N_liq_logMAE", "Traj_RMSE_balanced", "Brier", "one_minus_AUPRC", "Physics_Violation_Rate"],
    "probabilistic": ["N_liq_logMAE", "Traj_CRPS", "Brier", "Calibration_Error",
                      "one_minus_AUPRC", "Physics_Violation_Rate"],
    "physics": ["N_liq_logMAE", "__TRAJ__", "CRR_RMSE", "Brier", "Calibration_Error",
                "Physics_Violation_Rate"],
}
_SCORE_COLS = {
    "core": ("P3_Core_Raw_Score", "P3_Core_Admissible_Score"),
    "probabilistic": ("P3_Probabilistic_Raw_Score", "P3_Probabilistic_Admissible_Score"),
    "physics": ("P3_Physics_Raw_Score", "P3_Physics_Admissible_Score"),
}
P3_PROTOCOL_VERSION = "p3-prefix-conditioned-v3-strict-physics"
"""Версия заранее заданного профиля P³-весов/ворот для отчётных таблиц."""
# Epsilon-доминирование: микроразличия не делают модель Pareto-оптимальной.
_DEFAULT_EPS: Dict[str, float] = {
    "N_liq_logMAE": 0.02, "Traj_RMSE": 0.002, "Traj_RMSE_balanced": 0.002, "Brier": 0.001,
    "one_minus_AUPRC": 0.001, "Physics_Violation_Rate": 0.005, "Calibration_Error": 0.005,
    "Traj_CRPS": 0.002, "CRR_RMSE": 0.002,
}
_OBJ_PREFIX = "_pobj__"

# Gate компетентности: чтобы попасть в admissible-ранжирование, модель должна быть в пределах
# фактора от лучшего значения по КАЖДОЙ ключевой инженерной оси. Это ловит две патологии,
# которые score сам по себе пропускает: (1) коллапс на целом режиме опыта (через worst-state
# траекторию) и (2) катастрофу по числу циклов до разжижения N_liq, замаскированную сильными
# вспомогательными метриками (как у чисто-траекторного Transformer). Порог = factor·best.
_COMPETENCE_AXES: Dict[str, Dict[str, float]] = {
    "core": {"__WORST_TRAJ__": 3.0, "N_liq_logMAE": 3.0},
    "probabilistic": {"__WORST_TRAJ__": 3.0, "N_liq_logMAE": 3.0},
    "physics": {"__WORST_TRAJ__": 3.0, "N_liq_logMAE": 3.0, "CRR_RMSE": 3.0},
}


def metric_direction(metric_key: str) -> str:
    """
    Вернуть направление метрики: ``"min"`` (lower_is_better=True), ``"max"`` (lower_is_better=False)
    или ``"target"`` (если ``METRICS[metric_key].target`` не None). Неизвестные ключи → "min".

    :param metric_key: ключ метрики (для производного ``one_minus_AUPRC`` — направление "min")
    :return: "min" | "max" | "target"
    """
    if metric_key == "one_minus_AUPRC":
        return "min"
    info = METRICS.get(metric_key)
    if info is None:
        return "min"
    if info.target is not None:
        return "target"
    return "min" if info.lower_is_better else "max"


def compute_physical_admissibility(physics_violation_rate: float, soft_threshold: float = 0.0,
                                   hard_threshold: float = 0.0,
                                   penalty_strength: float = 3.0) -> Tuple[bool, float, float]:
    """
    Оценить физическую допустимость модели по доле физических нарушений.

    Логика:
    По умолчанию admissibility строгая: PVR=0 → gate 1; любая нарушившая контракт траектория
    (PVR>0) → gate 0. Ненулевые soft/hard thresholds поддерживаются только для sensitivity analysis.
    - PVR = NaN/inf → физически ненадёжна, gate 0.

    :param physics_violation_rate: доля физически невозможных кривых
    :return: кортеж (physically_unreliable, physical_penalty, physical_gate)
    """
    pvr = physics_violation_rate
    if pvr is None or pd.isna(pvr) or math.isinf(float(pvr)):
        return True, float("nan"), 0.0
    pvr = float(pvr)
    if pvr <= soft_threshold:
        return False, 0.0, 1.0
    if hard_threshold <= soft_threshold:
        return True, float(penalty_strength * pvr), 0.0
    penalty = penalty_strength * max(0.0, pvr - soft_threshold) / (hard_threshold - soft_threshold)
    if pvr <= hard_threshold:
        return False, float(penalty), 1.0
    return True, float(penalty), 0.0


def _physics_traj_col(df: pd.DataFrame) -> str:
    """Траекторная метрика для physics-режима: Traj_CRPS при наличии, иначе post-prefix/full RMSE."""
    if "Traj_CRPS" in df.columns and df["Traj_CRPS"].notna().any():
        return "Traj_CRPS"
    if "Traj_RMSE_continuation" in df.columns and df["Traj_RMSE_continuation"].notna().any():
        return "Traj_RMSE_continuation"
    return "Traj_RMSE"


def _preferred_balanced_traj_col(df: pd.DataFrame) -> str:
    """Вернуть primary trajectory metric for prefix-conditioned forecasting."""
    if ("Traj_RMSE_continuation_balanced" in df.columns
            and df["Traj_RMSE_continuation_balanced"].notna().any()):
        return "Traj_RMSE_continuation_balanced"
    if "Traj_RMSE_balanced" in df.columns and df["Traj_RMSE_balanced"].notna().any():
        return "Traj_RMSE_balanced"
    if "Traj_RMSE_continuation" in df.columns and df["Traj_RMSE_continuation"].notna().any():
        return "Traj_RMSE_continuation"
    return "Traj_RMSE"


def _preferred_worst_traj_col(df: pd.DataFrame) -> str:
    """Вернуть worst-state trajectory metric, preferring post-prefix continuation."""
    if ("Traj_RMSE_continuation_worst" in df.columns
            and df["Traj_RMSE_continuation_worst"].notna().any()):
        return "Traj_RMSE_continuation_worst"
    if "Traj_RMSE_worst" in df.columns and df["Traj_RMSE_worst"].notna().any():
        return "Traj_RMSE_worst"
    return _preferred_balanced_traj_col(df)


def _resolve_weights(df: pd.DataFrame, mode: str) -> Dict[str, float]:
    """Подставить реальную траекторную метрику вместо плейсхолдеров (__TRAJ__ / balanced fallback)."""
    traj = _physics_traj_col(df)
    balanced = _preferred_balanced_traj_col(df)
    out: Dict[str, float] = {}
    for k, v in _P3_WEIGHTS[mode].items():
        key = traj if k == "__TRAJ__" else k
        if key == "Traj_RMSE_balanced":
            key = balanced
        out[key] = v
    return out


def compute_competence(df: pd.DataFrame, mode: str = "core") -> pd.DataFrame:
    """
    Отметить «некомпетентные» модели — катастрофичные хотя бы по одной ключевой оси.

    Для каждой оси из :data:`_COMPETENCE_AXES` порог = ``factor · best`` (best — лучшее значение
    среди моделей с непустой метрикой; все оси «меньше — лучше»). Модель, превысившая порог хотя
    бы по одной оси, помечается ``competence_failed=True`` и исключается из admissible-ранжирования
    (но остаётся в таблице для диагностики). Оси с отсутствующей колонкой/значением пропускаются —
    модель не штрафуется за неумение, которое к ней неприменимо.

    Добавляет колонки ``competence_failed`` (bool) и ``competence_reason`` (str).

    :param df: таблица метрик (после compute_metrics)
    :param mode: режим P³ (определяет набор осей)
    :return: копия df с колонками компетентности
    """
    out = df.copy()
    axes = {
        (_preferred_worst_traj_col(out) if axis == "__WORST_TRAJ__" else axis): factor
        for axis, factor in _COMPETENCE_AXES.get(mode, _COMPETENCE_AXES["core"]).items()
    }
    failed = pd.Series(False, index=out.index)
    reasons = ["" for _ in range(len(out))]
    for axis, factor in axes.items():
        if axis not in out.columns:
            continue
        vals = pd.to_numeric(out[axis], errors="coerce")
        elig = vals.notna()
        if not elig.any():
            continue
        best = float(vals[elig].min())
        thresh = factor * best
        bad = elig & (vals > thresh)
        for pos, is_bad in enumerate(bad.to_numpy()):
            if is_bad:
                reasons[pos] = (reasons[pos] + "; " if reasons[pos] else "") + \
                    f"{axis}={float(vals.iloc[pos]):.3f}>{factor:g}×best({best:.3f})"
        failed = failed | bad
    out["competence_failed"] = failed.to_numpy()
    out["competence_reason"] = reasons
    return out


def compute_p3_score(df: pd.DataFrame, reference_model: str, mode: str = "core",
                     eps: float = 1e-9) -> pd.DataFrame:
    """
    Посчитать raw и physically-admissible P³-Score, нормированные к фиксированной reference-модели.

    Для «меньше — лучше»: ρ=(m+eps)/(ref+eps); для «больше — лучше»: ρ=(ref+eps)/(m+eps).
    ``P3_loss_raw = Σ w_j·ln(ρ_j)``; ``P3_raw_score = 100·exp(−P3_loss_raw)``.
    Физическая допустимость: ``P3_admissible_loss = P3_loss_raw + physical_penalty``;
    ``P3_admissible_score = 100·exp(−P3_admissible_loss)·physical_gate`` (0, если physically_unreliable).

    Добавляет ТОЛЬКО колонки выбранного режима + physically_unreliable / physical_penalty / physical_gate.
    NaN не заполняются: модель без обязательной метрики получает NaN-score. Для physics учитываются
    только модели с ``Produces_CRR == True`` и непустым ``CRR_RMSE``.

    :raises ValueError: если reference отсутствует или у него NaN в обязательной метрике.
    """
    if mode not in _P3_WEIGHTS:
        raise ValueError(f"Неизвестный режим '{mode}'. Допустимо: {sorted(_P3_WEIGHTS)}.")
    out = df.copy()
    if "model" not in out.columns:
        raise ValueError("В таблице нет колонки 'model'.")
    if reference_model not in set(out["model"]):
        raise ValueError(f"Reference-модель '{reference_model}' отсутствует в таблице.")
    weights = _resolve_weights(df, mode)
    missing = [m for m in weights if m not in out.columns]
    if missing:
        raise ValueError(f"В таблице отсутствуют колонки для режима '{mode}': {missing}.")
    ref = out.loc[out["model"] == reference_model].iloc[0]

    # Вырожденный AUPRC (один класс) — пометить и исключить AUPRC с перенормировкой весов.
    classification_unavailable = False
    if "AUPRC" in weights and pd.isna(ref.get("AUPRC", np.nan)):
        classification_unavailable = True
        weights = {k: v for k, v in weights.items() if k != "AUPRC"}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    if mode == "physics" and (not bool(ref.get("Produces_CRR", False)) or pd.isna(ref.get("CRR_RMSE", np.nan))):
        raise ValueError(f"Reference '{reference_model}' для physics-режима должен иметь Produces_CRR=True и CRR_RMSE.")
    for m in weights:
        if pd.isna(ref.get(m, np.nan)):
            raise ValueError(f"У reference-модели '{reference_model}' NaN в обязательной метрике '{m}'.")

    raw_col, adm_col = _SCORE_COLS[mode]
    raws, adms, unrel, pens, gates = [], [], [], [], []
    for _, row in out.iterrows():
        u, pen, gate = compute_physical_admissibility(row.get("Physics_Violation_Rate", np.nan))
        unrel.append(u); pens.append(pen); gates.append(gate)
        eligible = all(pd.notna(row.get(m, np.nan)) for m in weights)
        if mode == "physics":
            eligible = eligible and bool(row.get("Produces_CRR", False)) and pd.notna(row.get("CRR_RMSE", np.nan))
        if not eligible:
            raws.append(float("nan")); adms.append(float("nan")); continue
        loss = 0.0
        for m, w in weights.items():
            mv = float(row[m]); rv = float(ref[m])
            rho = (rv + eps) / (mv + eps) if metric_direction(m) == "max" else (mv + eps) / (rv + eps)
            loss += w * math.log(max(rho, 1e-12))
        raws.append(100.0 * math.exp(-loss))
        adms.append(0.0 if u else 100.0 * math.exp(-(loss + pen)) * gate)

    out[raw_col] = raws
    out[adm_col] = adms
    out["physically_unreliable"] = unrel
    out["physical_penalty"] = pens
    out["physical_gate"] = gates
    out.attrs["classification_component_unavailable"] = classification_unavailable
    return out


def build_pareto_objectives(df: pd.DataFrame, mode: str = "core",
                            admissible_only: bool = True) -> pd.DataFrame:
    """
    Построить objective-колонки Pareto (все «меньше — лучше»; AUPRC → 1 − AUPRC).

    NaN не заполняются. При ``admissible_only=True`` физически ненадёжные модели помечаются
    ``excluded_from_admissible_ranking=True`` (но остаются в таблице). Имена objective-колонок и
    карта eps сохраняются в ``result.attrs``.
    """
    if mode not in _P3_PARETO:
        raise ValueError(f"Неизвестный режим '{mode}'. Допустимо: {sorted(_P3_PARETO)}.")
    out = df.copy()
    bases = []
    for b in _P3_PARETO[mode]:
        if b == "__TRAJ__":
            bases.append(_physics_traj_col(df))
        elif b == "Traj_RMSE_balanced":
            bases.append(_preferred_balanced_traj_col(df))
        else:
            bases.append(b)
    obj_cols = []
    for base in bases:
        col = _OBJ_PREFIX + base
        if base == "one_minus_AUPRC":
            if "AUPRC" not in out.columns:
                raise ValueError("Нет колонки 'AUPRC' для objective one_minus_AUPRC.")
            out[col] = 1.0 - out["AUPRC"]
        else:
            if base not in out.columns:
                raise ValueError(f"Нет колонки '{base}' для режима '{mode}'.")
            out[col] = out[base]
        obj_cols.append(col)

    if "physically_unreliable" in out.columns:
        unreliable = out["physically_unreliable"].astype(bool)
    else:
        unreliable = out.get("Physics_Violation_Rate", pd.Series([np.nan] * len(out))).apply(
            lambda v: compute_physical_admissibility(v)[0])
    out["excluded_from_admissible_ranking"] = unreliable if admissible_only else False
    out.attrs["pareto_objectives"] = obj_cols
    out.attrs["pareto_mode"] = mode
    return out


def pareto_rank(df: pd.DataFrame, objective_cols: List[str],
                eps_by_metric: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Недоминируемая сортировка (nondominated sorting) с epsilon-доминированием.

    A доминирует B, если A не хуже B по всем objective (с допуском eps) и строго лучше хотя бы по
    одному (за пределами eps). Все objective — «меньше — лучше». Строки с NaN в любом objective не
    участвуют (front = NaN).

    Добавляет ``pareto_front_raw``, ``dominates_count``, ``dominated_count``.
    """
    out = df.copy()
    eps_by_metric = _DEFAULT_EPS if eps_by_metric is None else eps_by_metric
    eps_vec = np.array([eps_by_metric.get(c.replace(_OBJ_PREFIX, ""), 0.0) for c in objective_cols], dtype=float)
    n = len(out)
    vals = out[objective_cols].to_numpy(dtype=float)
    valid = ~np.isnan(vals).any(axis=1)
    idx = np.where(valid)[0]

    def dominates(a, b):
        return bool(np.all(vals[a] <= vals[b] + eps_vec) and np.any(vals[a] < vals[b] - eps_vec))

    dominates_count = np.zeros(n, dtype=int)
    dominated_count = np.zeros(n, dtype=int)
    dom_list = {i: [] for i in idx}
    dom_by = {i: 0 for i in idx}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            if dominates(i, j):
                dom_list[i].append(j); dominates_count[i] += 1
            elif dominates(j, i):
                dominated_count[i] += 1; dom_by[i] += 1

    front = np.full(n, np.nan)
    current = [i for i in idx if dom_by[i] == 0]
    remaining = dict(dom_by)
    front_no = 1
    while current:
        for i in current:
            front[i] = front_no
        nxt = []
        for i in current:
            for j in dom_list[i]:
                remaining[j] -= 1
                if remaining[j] == 0:
                    nxt.append(j)
        current = nxt
        front_no += 1

    out["pareto_front_raw"] = front
    out["dominates_count"] = dominates_count
    out["dominated_count"] = dominated_count
    return out


def publication_ranking_table(df: pd.DataFrame, reference_model: str, mode: str = "core") -> pd.DataFrame:
    """
    Итоговая публикационная таблица: raw + admissible P³-Score и raw + admissible Pareto-фронты.

    Сортировка: ``excluded_from_admissible_ranking`` ASC → ``pareto_front_admissible`` ASC →
    ``P3_*_Admissible_Score`` DESC → ``Physics_Violation_Rate`` ASC. Главный публикационный результат —
    admissible-фронт и admissible-score; raw-версии диагностические.
    """
    scored = compute_p3_score(df, reference_model, mode)
    scored = compute_competence(scored, mode) # gate компетентности (worst-режим + N_liq)
    raw_col, adm_col = _SCORE_COLS[mode]

    # admissible-score обнуляется и для некомпетентных моделей (не только физически ненадёжных)
    scored.loc[scored["competence_failed"].astype(bool), adm_col] = 0.0

    # raw Pareto (диагностический, без исключения ненадёжных)
    raw_obj = build_pareto_objectives(scored, mode, admissible_only=False)
    ranked = pareto_rank(raw_obj, raw_obj.attrs["pareto_objectives"])

    # admissible Pareto: исключить физически ненадёжные И некомпетентные
    adm = build_pareto_objectives(ranked, mode, admissible_only=True)
    adm["excluded_from_admissible_ranking"] = (adm["excluded_from_admissible_ranking"].astype(bool)
                                               | adm["competence_failed"].astype(bool))
    obj_cols = adm.attrs["pareto_objectives"]
    keep = ~adm["excluded_from_admissible_ranking"].astype(bool)
    adm["pareto_front_admissible"] = np.nan
    if keep.any():
        sub = pareto_rank(adm.loc[keep].copy(), obj_cols)
        adm.loc[keep, "pareto_front_admissible"] = sub["pareto_front_raw"].to_numpy()

    adm = adm.sort_values(
        by=["excluded_from_admissible_ranking", "pareto_front_admissible", adm_col, "Physics_Violation_Rate"],
        ascending=[True, True, False, True], na_position="last",
    ).reset_index(drop=True)
    adm.attrs["classification_component_unavailable"] = scored.attrs.get("classification_component_unavailable", False)

    cols_by_mode = {
        "core": ["model", "pareto_front_raw", "pareto_front_admissible", raw_col, adm_col,
                 "physically_unreliable", "competence_failed", "competence_reason",
                 "excluded_from_admissible_ranking", "physical_penalty",
                 "Physics_Violation_Rate", "N_liq_logMAE", "Traj_RMSE_continuation_balanced",
                 "Traj_RMSE_continuation_worst", "Traj_RMSE_continuation",
                 "Traj_RMSE_balanced", "Traj_RMSE_worst",
                 "Traj_RMSE_liq", "Traj_RMSE_stab", "Traj_RMSE_nostab", "Traj_RMSE", "Brier", "AUPRC",
                 "N_liq_MAE", "N_liq_RMSE", "AUROC", "ECE", "Traj_MAE", "Traj_MSE", "Produces_CRR"],
        "probabilistic": ["model", "pareto_front_raw", "pareto_front_admissible", raw_col, adm_col,
                          "physically_unreliable", "competence_failed", "competence_reason",
                          "excluded_from_admissible_ranking", "physical_penalty",
                          "Physics_Violation_Rate", "N_liq_logMAE", "Traj_CRPS",
                          "Traj_RMSE_continuation_worst", "Traj_RMSE_worst", "Brier",
                          "Calibration_Error", "AUPRC", "Coverage_80", "Coverage_90", "Coverage_95",
                          "Interval_Width_90", "Traj_NLL", "ECE"],
        "physics": ["model", "pareto_front_raw", "pareto_front_admissible", raw_col, adm_col,
                    "physically_unreliable", "competence_failed", "competence_reason",
                    "excluded_from_admissible_ranking", "physical_penalty",
                    "Physics_Violation_Rate", "N_liq_logMAE", "Traj_CRPS",
                    "Traj_RMSE_continuation_worst", "Traj_RMSE_worst", "CRR_RMSE", "Brier",
                    "Calibration_Error", "Produces_CRR"],
    }
    cols = [c for c in cols_by_mode[mode] if c in adm.columns]
    result = adm[cols].copy()
    result.insert(1, "P3_protocol_version", P3_PROTOCOL_VERSION)
    result.attrs["classification_component_unavailable"] = adm.attrs["classification_component_unavailable"]
    return result


# --- регистрация новых метрик в каталоге и локализации колонок ---
METRICS["P3_Core_Raw_Score"] = MetricInfo(
    "P3_Core_Raw_Score", "P³ Core raw score",
    "Raw Predictive–Probabilistic–Physical score normalized against a fixed reference model. This "
    "score does not apply the hard physical admissibility gate.",
    "–", lower_is_better=False, fmt=".1f")
METRICS["P3_Core_Admissible_Score"] = MetricInfo(
    "P3_Core_Admissible_Score", "P³ Core admissible score",
    "Physically admissible Predictive–Probabilistic–Physical score. It combines event-timing accuracy, "
    "trajectory accuracy, probabilistic risk quality and physical admissibility. Models above the hard "
    "physical violation threshold receive a score of zero.",
    "–", lower_is_better=False, fmt=".1f")
METRICS["P3_Probabilistic_Raw_Score"] = MetricInfo(
    "P3_Probabilistic_Raw_Score", "P³ Probabilistic raw score",
    "Raw probabilistic P³ score (event timing, probabilistic trajectory CRPS, risk Brier, calibration) "
    "without the hard physical admissibility gate.",
    "–", lower_is_better=False, fmt=".1f")
METRICS["P3_Probabilistic_Admissible_Score"] = MetricInfo(
    "P3_Probabilistic_Admissible_Score", "P³ Probabilistic admissible score",
    "Physically admissible probabilistic P³ score based on event timing, probabilistic trajectory "
    "quality, risk probability quality, calibration and physical consistency.",
    "–", lower_is_better=False, fmt=".1f")
METRICS["P3_Physics_Raw_Score"] = MetricInfo(
    "P3_Physics_Raw_Score", "P³ Physics raw score",
    "Raw physics-only P³ score for CRR-capable models, without the hard physical admissibility gate.",
    "–", lower_is_better=False, fmt=".1f")
METRICS["P3_Physics_Admissible_Score"] = MetricInfo(
    "P3_Physics_Admissible_Score", "P³ Physics admissible score",
    "Physics-only admissible P³ score for models that explicitly produce CRR curves. It must only be "
    "computed for CRR-capable models.",
    "–", lower_is_better=False, fmt=".1f")
METRICS["pareto_front_raw"] = MetricInfo(
    "pareto_front_raw", "Pareto front (raw)",
    "Raw non-dominated sorting rank computed from selected predictive, probabilistic and physical "
    "objectives before excluding physically unreliable models.",
    "–", lower_is_better=True, target=1.0, fmt=".0f")
METRICS["pareto_front_admissible"] = MetricInfo(
    "pareto_front_admissible", "Pareto front (admissible)",
    "Non-dominated sorting rank after applying the physical admissibility gate. Lower front number is better.",
    "–", lower_is_better=True, target=1.0, fmt=".0f")
METRICS["physically_unreliable"] = MetricInfo(
    "physically_unreliable", "Physically unreliable",
    "Boolean flag indicating that the model exceeds the hard physical violation threshold and should "
    "not be considered admissible for publication ranking.",
    "–", lower_is_better=True)
METRICS["physical_penalty"] = MetricInfo(
    "physical_penalty", "Physical penalty",
    "Soft penalty added to P³ loss for small but non-negligible physical violation rates.",
    "–", lower_is_better=True)
METRICS["excluded_from_admissible_ranking"] = MetricInfo(
    "excluded_from_admissible_ranking", "Excluded from admissible ranking",
    "Boolean flag indicating that the model is shown for diagnostics but excluded from admissible "
    "Pareto ranking.",
    "–", lower_is_better=True)
METRICS["competence_failed"] = MetricInfo(
    "competence_failed", "Competence gate failed",
    "Boolean flag: the model is catastrophic on at least one key engineering axis (preferred "
    "worst-state trajectory RMSE — post-prefix when available — or N_liq log-MAE more than a "
    "factor worse than the best model), so it is excluded from admissible ranking even if it is "
    "physically consistent.",
    "–", lower_is_better=True)

METRIC_COLUMN_EN.update({
    "P3_Core_Raw_Score": "P³ Core raw", "P3_Core_Admissible_Score": "P³ Core admissible",
    "P3_Probabilistic_Raw_Score": "P³ Prob. raw", "P3_Probabilistic_Admissible_Score": "P³ Prob. admissible",
    "P3_Physics_Raw_Score": "P³ Physics raw", "P3_Physics_Admissible_Score": "P³ Physics admissible",
    "pareto_front_raw": "Pareto front (raw)", "pareto_front_admissible": "Pareto front (adm.)",
    "physically_unreliable": "Physically unreliable", "physical_penalty": "Physical penalty",
    "excluded_from_admissible_ranking": "Excluded (adm.)",
    "competence_failed": "Competence gate failed", "competence_reason": "Competence gate reason",
})
