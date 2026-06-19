"""
Многосидовая оценка с доверительными интервалами (замечание рецензента P1).

Для нескольких сидов одновременно меняется и **разбиение** train/val/test (стратификация
зависит от ``config.seed``), и **инициализация** обучения — это честная оценка дисперсии по
повторным сгруппированным сплитам, а не один случайный split. Обучаются три предложенные
физические модели и опорная PINN по тому же протоколу, что и в основных ноутбуках
(physics → косинусный LR, PINN → baseline-эпохи).

Запуск по одному сиду (инкрементально дописывает results/analysis_tables/multiseed_raw.csv):
    python run_multiseed.py --seed 42
    python run_multiseed.py --seed 1
    ...
Затем агрегирование в сводку со средними ± 95% ДИ и фигуру:
    python run_multiseed.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

from liquefaction_ai import (get_default_config, load_population_artifact,  # noqa: E402
                             prepare_benchmark_dataset, train_model)
from liquefaction_ai.config import set_global_seed  # noqa: E402
from liquefaction_ai.evaluation import collect_outputs, compute_metrics  # noqa: E402
from liquefaction_ai.evaluation.p3_ranking import compute_p3_score  # noqa: E402
from liquefaction_ai.training.persistence import load_model_metadata  # noqa: E402
from liquefaction_ai import models as M  # noqa: E402

TABLES = os.path.join(REPO, "results", "analysis_tables")
ANALYSIS_FIGS = os.path.join(REPO, "results", "analysis_figs")
DATA = os.path.join(REPO, "data", "demo_run")
GROUPED = False   # переопределяется флагом --grouped (leakage-free разбиение по объекту)


def _raw_path() -> str:
    return os.path.join(TABLES, "multiseed_grouped_raw.csv" if GROUPED else "multiseed_raw.csv")
# (имя_артефакта, отображаемое_имя, физическая_ли_модель)
MODELS = [("dpi_flow", "DPI-Flow", True), ("evt_ssm", "EVT-NeuralSSM", True),
          ("dpi_evt", "DPI-EVT", True), ("pinn", "PINN", False)]
METRIC_KEYS = ["N_liq_logMAE", "Traj_RMSE", "AUROC", "Brier", "Physics_Violation_Rate"]


def run_seed(seed: int) -> None:
    device = torch.device("cpu")
    pop, config = load_population_artifact(DATA)
    config.seed = seed                      # меняет и сплит (стратификация), и инициализацию
    config.group_split_by_object = GROUPED  # leakage-free разбиение по объекту, если включено
    set_global_seed(seed)
    bench = prepare_benchmark_dataset(pop, config, device)
    train, val, test = bench["train"], bench["val"], bench["test"]
    if GROUPED:
        tr_o = set(train["meta"]["object"]); te_o = set(test["meta"]["object"])
        assert not (tr_o & te_o), "leakage: объект и в train, и в test"

    per_model_metrics = {}
    for name, disp, is_phys in MODELS:
        hp, _ = load_model_metadata(os.path.join(REPO, "models"), name)
        cls = getattr(M, hp["model_type"])
        set_global_seed(seed)
        model = cls(**hp["model_kwargs"]).to(device)
        epochs = config.physics_epochs if is_phys else config.baseline_epochs
        model, _ = train_model(model, train, val, epochs=epochs, model_name=f"{disp}(s{seed})",
                               config=config, device=device, verbose=False,
                               scheduler="cosine" if is_phys else "none")
        met, _ = compute_metrics(disp, collect_outputs(model, test, config, device), test, config)
        per_model_metrics[disp] = met

    # P³-Core (core) с reference=PINN на этом сплите
    df = pd.DataFrame([{**{"model": d}, **{k: per_model_metrics[d].get(k) for k in
                        ["N_liq_logMAE", "Traj_RMSE", "Brier", "AUPRC", "Physics_Violation_Rate"]}}
                       for d in per_model_metrics])
    try:
        scored = compute_p3_score(df, "PINN", "core")
        p3 = dict(zip(scored["model"], scored["P3_Core_Raw_Score"]))
    except Exception:
        p3 = {d: float("nan") for d in per_model_metrics}

    rows = []
    for disp, met in per_model_metrics.items():
        row = {"seed": seed, "model": disp, "P3_Core": p3.get(disp, float("nan"))}
        row.update({k: met.get(k) for k in METRIC_KEYS})
        rows.append(row)
    out = pd.DataFrame(rows)
    raw = _raw_path()
    os.makedirs(TABLES, exist_ok=True)
    header = not os.path.exists(raw)
    out.to_csv(raw, mode="a", header=header, index=False)
    print(f"seed {seed}{' [grouped]' if GROUPED else ''}: дописано {len(out)} строк → {raw}")
    print(out[["model", "P3_Core"] + METRIC_KEYS].round(4).to_string(index=False))


def aggregate() -> None:
    from liquefaction_ai.viz import new_figure, save_figure, QUALITATIVE, INK

    raw = pd.read_csv(_raw_path())
    suffix = "_grouped" if GROUPED else ""
    tag = " [leakage-free, grouped by object]" if GROUPED else ""
    n_seeds = raw["seed"].nunique()
    cols = ["P3_Core"] + METRIC_KEYS
    agg = raw.groupby("model")[cols].agg(["mean", "std", "count"])
    rows = []
    for model in agg.index:
        rec = {"model": model, "n_seeds": int(agg.loc[model, ("P3_Core", "count")])}
        for c in cols:
            mean = float(agg.loc[model, (c, "mean")])
            std = float(agg.loc[model, (c, "std")]) if agg.loc[model, (c, "count")] > 1 else 0.0
            ci = 1.96 * std / np.sqrt(max(int(agg.loc[model, (c, "count")]), 1))
            rec[f"{c}_mean"] = round(mean, 4)
            rec[f"{c}_ci95"] = round(ci, 4)
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values("P3_Core_mean", ascending=False).reset_index(drop=True)
    os.makedirs(TABLES, exist_ok=True)
    summary.to_csv(os.path.join(TABLES, f"multiseed{suffix}_summary.csv"), index=False)
    print(f"=== Многосидовая сводка ({n_seeds} сидов){tag} ===")
    show = ["model", "n_seeds", "P3_Core_mean", "P3_Core_ci95",
            "N_liq_logMAE_mean", "N_liq_logMAE_ci95", "Traj_RMSE_mean", "Traj_RMSE_ci95"]
    print(summary[show].to_string(index=False))

    order = summary.sort_values("P3_Core_mean")["model"].tolist()
    means = [summary.set_index("model").loc[m, "P3_Core_mean"] for m in order]
    cis = [summary.set_index("model").loc[m, "P3_Core_ci95"] for m in order]
    figw, fig = new_figure((7.0, 4.0)); ax = fig.add_subplot(111)
    colors = ["#C36F6F" if m in ("DPI-Flow", "EVT-NeuralSSM", "DPI-EVT") else ("#E0A458" if m == "PINN" else "#7C9CB5") for m in order]
    ax.barh(order, means, xerr=cis, color=colors, edgecolor=INK, linewidth=0.6,
            error_kw=dict(ecolor=INK, capsize=4, lw=1.0))
    ax.axvline(100.0, ls="--", color="#9aa3ad", lw=1.0)
    ax.set_xlabel("P³-Core score (mean ± 95% CI across seeds)")
    ax.set_title(f"Многосидовая устойчивость P³ ({n_seeds} повторных сплитов){tag}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig_id = "3_6_multiseed_grouped_ci" if GROUPED else "3_5_multiseed_ci"
    save_figure(figw, fig_id, True, results_dir=ANALYSIS_FIGS)
    print(f"\nsaved results/analysis_tables/multiseed{suffix}_summary.csv и results/analysis_figs/{fig_id}.png/pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--grouped", action="store_true",
                    help="leakage-free разбиение по объекту (ни один объект не в train и test)")
    args = ap.parse_args()
    GROUPED = args.grouped
    if args.aggregate:
        aggregate()
    elif args.seed is not None:
        run_seed(args.seed)
    else:
        ap.error("укажите --seed S или --aggregate")
