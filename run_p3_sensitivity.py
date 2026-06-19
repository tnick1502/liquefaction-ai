"""
Анализ чувствительности публикационной метрики P³ (замечание рецензента P1).

Проверяет, что лидерство моделей в P³-рейтинге устойчиво к авторским проектным решениям:
1. **Веса** непересекающихся критериев — случайные возмущения симплекса весов (Дирихле);
2. **Пороги физического gate** (soft/hard) — сетка значений;
3. **Reference-модель** — нормировка к разным опорным моделям (PINN/DPI-Flow/Transformer);
4. **Вкл/выкл risk-only моделей** (CatBoost/FT-Transformer/MLP-Risk).

Для каждого возмущения считается ранжирование пригодных (non-NaN) моделей и сравнивается с
номинальным по Kendall-τ; фиксируется частота top-1. Результаты: results/analysis_tables/p3_sensitivity.csv
и фигура results/analysis_figs/3_4_p3_sensitivity.{png,pdf}. Скрипт post-hoc — переобучение не требуется.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from liquefaction_ai.evaluation import p3_ranking as P3  # noqa: E402
from liquefaction_ai.viz import new_figure, save_figure, QUALITATIVE, INK  # noqa: E402

TABLES = os.path.join(REPO, "results", "tables")
ANALYSIS_TABLES = os.path.join(REPO, "results", "analysis_tables")
ANALYSIS_FIGS = os.path.join(REPO, "results", "analysis_figs")
MODE = "core"
NOMINAL_REF = "PINN"
RISK_ONLY = {"CatBoost", "FT-Transformer", "MLP-Risk"}
N_PERTURB = 600
SEED = 42


def eligible_ranking(df: pd.DataFrame, reference: str, weights=None, gate=None) -> list:
    """Вернуть список моделей по убыванию admissible P³-score под заданными весами/gate."""
    saved_w = dict(P3._P3_WEIGHTS[MODE])
    saved_adm = P3.compute_physical_admissibility
    try:
        if weights is not None:
            P3._P3_WEIGHTS[MODE] = weights
        if gate is not None:
            soft, hard = gate
            P3.compute_physical_admissibility = (
                lambda pvr, soft_threshold=soft, hard_threshold=hard, penalty_strength=3.0:
                saved_adm(pvr, soft_threshold, hard_threshold, penalty_strength))
        table = P3.publication_ranking_table(df, reference, MODE)
    finally:
        P3._P3_WEIGHTS[MODE] = saved_w
        P3.compute_physical_admissibility = saved_adm
    col = "P3_Core_Admissible_Score"
    ok = table[table[col].notna() & (table[col] > 0)]
    return ok.sort_values(col, ascending=False)["model"].tolist()


def kendall_tau(rank_a: list, rank_b: list) -> float:
    """Kendall-τ между двумя ранжированиями общих моделей."""
    common = [m for m in rank_a if m in rank_b]
    if len(common) < 2:
        return float("nan")
    pos_a = {m: i for i, m in enumerate(rank_a)}
    pos_b = {m: i for i, m in enumerate(rank_b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            s = np.sign(pos_a[x] - pos_a[y]) * np.sign(pos_b[x] - pos_b[y])
            conc += s > 0
            disc += s < 0
    return (conc - disc) / (conc + disc) if (conc + disc) else float("nan")


def main() -> None:
    lb = pd.read_csv(os.path.join(TABLES, "full_leaderboard.csv"))
    rng = np.random.default_rng(SEED)
    nominal = eligible_ranking(lb, NOMINAL_REF)
    top1_nominal = nominal[0]
    print(f"Номинальный рейтинг ({len(nominal)} пригодных моделей): {nominal}")
    print(f"Номинальный лидер: {top1_nominal}")

    keys = list(P3._P3_WEIGHTS[MODE].keys())
    w_nom = np.array([P3._P3_WEIGHTS[MODE][k] for k in keys])
    rows = []

    # 1) Возмущение весов (Дирихле вокруг номинала, концентрация даёт ±~40% разброс)
    taus, top1_hits = [], 0
    conc = 12.0
    for _ in range(N_PERTURB):
        w = rng.dirichlet(conc * w_nom / w_nom.sum())
        ranking = eligible_ranking(lb, NOMINAL_REF, weights=dict(zip(keys, w)))
        taus.append(kendall_tau(nominal, ranking))
        top1_hits += int(ranking and ranking[0] == top1_nominal)
    taus = np.array([t for t in taus if not np.isnan(t)])
    rows.append({"test": "weights (Dirichlet, n=%d)" % N_PERTURB, "top1_stable_%": 100.0 * top1_hits / N_PERTURB,
                 "kendall_tau_mean": float(taus.mean()), "kendall_tau_min": float(taus.min()),
                 "note": f"лидер={top1_nominal}"})

    # 2) Пороги физического gate
    gate_stable = 0; gate_total = 0; gate_top1 = set()
    for soft in (0.005, 0.01, 0.02):
        for hard in (0.02, 0.05, 0.10, 0.20):
            if hard <= soft:
                continue
            r = eligible_ranking(lb, NOMINAL_REF, gate=(soft, hard))
            gate_total += 1; gate_stable += int(r and r[0] == top1_nominal); gate_top1.add(r[0] if r else None)
    rows.append({"test": "gate thresholds (soft×hard grid)", "top1_stable_%": 100.0 * gate_stable / gate_total,
                 "kendall_tau_mean": float("nan"), "kendall_tau_min": float("nan"),
                 "note": "top1 ∈ %s" % sorted(x for x in gate_top1 if x)})

    # 3) Reference-инвариантность
    refs = [m for m in ("PINN", "DPI-Flow", "Transformer", "DPI-EVT") if m in set(lb["model"])]
    ref_rankings = {r: eligible_ranking(lb, r) for r in refs}
    ref_identical = all(ref_rankings[r] == nominal for r in refs)
    rows.append({"test": "reference model (%s)" % ", ".join(refs),
                 "top1_stable_%": 100.0 if all((ref_rankings[r][0] == top1_nominal) for r in refs) else 0.0,
                 "kendall_tau_mean": 1.0 if ref_identical else float("nan"),
                 "kendall_tau_min": 1.0 if ref_identical else float("nan"),
                 "note": "ранжирование идентично для всех reference" if ref_identical else "зависит от reference"})

    # 4) Вкл/выкл risk-only моделей
    lb_no_risk = lb[~lb["model"].isin(RISK_ONLY)].copy()
    r_no_risk = eligible_ranking(lb_no_risk, NOMINAL_REF)
    rows.append({"test": "exclude risk-only (%s)" % ", ".join(sorted(RISK_ONLY)),
                 "top1_stable_%": 100.0 if (r_no_risk and r_no_risk[0] == top1_nominal) else 0.0,
                 "kendall_tau_mean": kendall_tau(nominal, r_no_risk), "kendall_tau_min": float("nan"),
                 "note": "ранжирование пригодных не меняется" if r_no_risk == [m for m in nominal if m not in RISK_ONLY] else "меняется"})

    summary = pd.DataFrame(rows)
    os.makedirs(ANALYSIS_TABLES, exist_ok=True)
    summary.to_csv(os.path.join(ANALYSIS_TABLES, "p3_sensitivity.csv"), index=False)
    print("\n=== P³ sensitivity ===")
    print(summary.to_string(index=False))

    # Фигура: (a) частота top-1 по тестам, (b) гистограмма Kendall-τ под возмущением весов
    figw, fig = new_figure((9.0, 3.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.barh(summary["test"], summary["top1_stable_%"], color=QUALITATIVE[0], edgecolor=INK, linewidth=0.5)
    ax1.set_xlim(0, 102); ax1.set_xlabel("top-1 stability, %")
    ax1.set_title(f"P³ top-1 = {top1_nominal}: устойчивость")
    ax1.invert_yaxis(); ax1.grid(axis="x", alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(taus, bins=20, color=QUALITATIVE[2], edgecolor="white")
    ax2.set_xlabel("Kendall-τ vs nominal (weight perturbations)"); ax2.set_ylabel("count")
    ax2.set_title("Стабильность ранга при возмущении весов")
    fig.tight_layout()
    save_figure(figw, "3_4_p3_sensitivity", True, results_dir=ANALYSIS_FIGS)
    print("\nsaved results/analysis_tables/p3_sensitivity.csv и results/analysis_figs/3_4_p3_sensitivity.png/pdf")


if __name__ == "__main__":
    main()
