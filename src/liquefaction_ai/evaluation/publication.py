"""
Publication figures and headline table (AAAI-focused metric emphasis).

Pure numpy/pandas/matplotlib — runs on cross-validation artifacts without torch. Used by notebook
3_7. All figure text is in ENGLISH (publication language). A clean, consistent palette is used so
the key results read as figures, not bare tables.

API
    reliability_diagram(samples_df, ref, bins, out_dir, suffix)        -> (fig, ece)
    forest_plot(cluster_df, metric, higher_better, ref, out_dir, ...)  -> fig   (model CI comparison)
    ablation_bars(ablation_summary, metric, out_dir, ...)              -> fig   (component contribution)
    pareto_plot(summary_df, out_dir, ...)                              -> fig   (onset vs trajectory)
    headline_table(summary_df, cluster_df)                            -> DataFrame
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Consistent palette (matches viz theme without importing the torch-bearing package)
INK = "#2b2f36"
ACCENT = "#C36F6F"       # proposed / highlighted model
AMBER = "#E0A458"        # reference (PINN)
BLUE = "#7C9CB5"         # other baselines
GRIDC = "#dfe3e8"
STRUCTURED = {"DPI-Flow", "DPI-EVT", "EVT-NeuralSSM"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": "#9aa3ad", "axes.grid": True,
                         "grid.color": GRIDC, "grid.linewidth": 0.7, "font.size": 10})
    return plt


def _bar_color(model: str, ref: str) -> str:
    if model == ref:
        return AMBER
    return ACCENT if model in STRUCTURED else BLUE


def reliability_diagram(samples_df: pd.DataFrame, ref: str = "DPI-Flow", bins: int = 10,
                        out_dir: Optional[str] = None, suffix: str = "grouped") -> Tuple[object, float]:
    """Reliability diagram of onset-risk calibration from per-sample (risk_prob_pred, liq_label)."""
    plt = _mpl()
    sub = samples_df[samples_df["model"] == ref]
    if sub.empty:
        raise ValueError(f"model {ref} not present in per-sample data")
    # ИСКЛЮЧАЕМ образцы с НЕнаблюдаемой меткой риска (незавершённые non-liq опыты): их нельзя
    # трактовать как известный отрицательный класс. Без фильтра reliability/ECE расходятся с таблицей
    # (метрики риска маскированы по risk_label_observed) — фигура противоречила бы leaderboard.
    risk_mask_col = "risk_label_observed" if "risk_label_observed" in sub.columns else "n_liq_observed"
    if risk_mask_col in sub.columns:
        sub = sub[sub[risk_mask_col].to_numpy() > 0.5]
        if sub.empty:
            raise ValueError(f"model {ref}: нет наблюдаемых меток риска после маскирования")
    y = (sub["liq_label"].to_numpy() > 0.5).astype(float)
    p = sub["risk_prob_pred"].to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns, ece = [], [], [], 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        conf, acc = p[m].mean(), y[m].mean()
        xs.append(conf); ys.append(acc); ns.append(int(m.sum()))
        ece += abs(conf - acc) * m.sum() / len(p)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="#9aa3ad", lw=1.0, label="Perfect calibration")
    sizes = 30 + 220 * np.array(ns) / max(max(ns), 1) if ns else []
    ax.scatter(xs, ys, s=sizes, color=ACCENT, edgecolor=INK, linewidth=0.6, zorder=3,
               label=f"{ref} (marker size ∝ count)")
    ax.plot(xs, ys, color=ACCENT, lw=1.2, alpha=0.6, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk (confidence)")
    ax.set_ylabel("Observed liquefaction frequency")
    ax.set_title(f"Reliability diagram — {ref} ({suffix})\nECE = {ece:.3f}")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"reliability_{ref}_{suffix}.{ext}"), dpi=150)
    return fig, float(ece)


def forest_plot(cluster_df: pd.DataFrame, metric: str = "AUPRC", higher_better: bool = True,
                ref: str = "DPI-Flow", title: Optional[str] = None,
                out_dir: Optional[str] = None, suffix: str = "grouped") -> object:
    """Dot-and-whisker plot of a metric across models with **object-cluster bootstrap 95% CI**.

    cluster_df must contain columns ``metric``, ``metric_lo``, ``metric_hi`` (from object_cluster_bootstrap).
    """
    plt = _mpl()
    d = cluster_df.dropna(subset=[metric]).copy()
    d = d.sort_values(metric, ascending=not higher_better)
    y = np.arange(len(d))
    point = d[metric].to_numpy()
    lo = d.get(f"{metric}_lo", point); hi = d.get(f"{metric}_hi", point)
    err = np.vstack([point - np.asarray(lo), np.asarray(hi) - point])
    colors = [_bar_color(m, ref) for m in d["model"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(d) + 1.4))
    ax.errorbar(point, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=2)
    ax.scatter(point, y, s=70, c=colors, edgecolor=INK, linewidth=0.6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(d["model"])
    ax.set_xlabel(f"{metric}  ({'higher' if higher_better else 'lower'} is better)  — 95% object-cluster bootstrap CI")
    ax.set_title(title or f"Model comparison: {metric} ({suffix})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"forest_{metric}_{suffix}.{ext}"), dpi=150)
    return fig


def ablation_bars(ablation_summary: pd.DataFrame, metric: str = "Traj_RMSE_worst",
                  baseline: str = "full", higher_better: bool = False,
                  out_dir: Optional[str] = None) -> object:
    """Component-contribution bars: a metric across ablation variants, 'full' highlighted.

    ablation_summary has columns ``ablation`` and ``<metric>_mean`` (+ optional ``<metric>_ci95``).
    """
    plt = _mpl()
    mcol = f"{metric}_mean" if f"{metric}_mean" in ablation_summary.columns else metric
    d = ablation_summary.dropna(subset=[mcol]).copy().sort_values(mcol, ascending=higher_better)
    vals = d[mcol].to_numpy()
    ci = d.get(f"{metric}_ci95", np.zeros(len(d)))
    colors = [ACCENT if a == baseline else BLUE for a in d["ablation"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(d) + 1.4))
    ax.barh(np.arange(len(d)), vals, xerr=np.asarray(ci), color=colors, edgecolor=INK, linewidth=0.6,
            error_kw=dict(ecolor=INK, capsize=3, lw=0.9))
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["ablation"])
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric}  ({'higher' if higher_better else 'lower'} is better)  [mean ± 95% CI over folds]")
    ax.set_title(f"Ablation: component contribution to {metric}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"ablation_{metric}.{ext}"), dpi=150)
    return fig


def pareto_plot(summary_df: pd.DataFrame, x: str = "Traj_RMSE_continuation", y: str = "AUPRC",
                ref: str = "DPI-Flow", out_dir: Optional[str] = None, suffix: str = "grouped") -> object:
    """Onset vs trajectory trade-off scatter — the 'competitive Pareto balance' story.

    x = post-prefix trajectory error (lower better), y = AUPRC onset (higher better). Physics-admissible
    models (Physics_Violation_Rate ≈ 0) are drawn as filled, others hollow; structured models accented.
    """
    plt = _mpl()
    xs_col, ys_col = f"{x}_mean", f"{y}_mean"
    d = summary_df.dropna(subset=[c for c in (xs_col, ys_col) if c in summary_df.columns]).copy()
    if xs_col not in d or ys_col not in d:
        raise ValueError(f"summary lacks {xs_col}/{ys_col}")
    pvr = d.get("Physics_Violation_Rate_mean", pd.Series(np.zeros(len(d)), index=d.index))
    gate = 0.05   # physical-feasibility gate of the P3 profile (PVR>gate => excluded/unreliable)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for _, r in d.iterrows():
        m = r["model"]; admissible = float(pvr.get(r.name, 0.0)) <= gate
        col = _bar_color(m, ref)
        ax.scatter(r[xs_col], r[ys_col], s=130, c=(col if admissible else "white"),
                   edgecolor=col if admissible else INK, linewidth=1.4, zorder=3)
        ax.annotate(m, (r[xs_col], r[ys_col]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.scatter([], [], s=90, c=BLUE, edgecolor=INK, label="physics-admissible (PVR≤0.05)")
    ax.scatter([], [], s=90, c="white", edgecolor=INK, label="physics-violating")
    ax.set_xlabel(f"{x} — post-prefix trajectory error (lower better)")
    ax.set_ylabel(f"{y} — onset detection (higher better)")
    ax.set_title(f"Onset vs trajectory trade-off ({suffix})\nupper-left = better; filled = feasible")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"pareto_{suffix}.{ext}"), dpi=150)
    return fig


# Headline metric order/labels (AUPRC + calibration + CRPS + per-state foregrounded; AUROC reference)
HEADLINE_ORDER = [
    ("AUPRC", "AUPRC ↑ (primary onset)"),
    ("Brier", "Brier ↓"),
    ("ECE", "ECE ↓ (calibration)"),
    ("Coverage_90", "Coverage@90 → 0.90"),
    ("Traj_CRPS", "PPR CRPS ↓ (proper)"),
    ("Traj_RMSE_continuation", "PPR RMSE ↓ (post-prefix)"),
    ("Traj_RMSE_continuation_worst", "PPR RMSE worst-state ↓"),
    ("N_liq_logMAE", "N_liq log-MAE ↓ (censored-aware)"),
    ("N_liq_logMAE_liq", "N_liq log-MAE ↓ (liquefied-only)"),
    ("CRR_RMSE", "CRR RMSE ↓ (DPI family)"),
    ("Physics_Violation_Rate", "Physics violations ↓"),
    ("AUROC", "AUROC ↑ (reference only)"),
]
# headline metric -> object-cluster bootstrap column (prefer cluster CI over naive fold CI)
_CLUSTER_CI_MAP = {
    "AUPRC": "AUPRC", "Brier": "Brier", "ECE": "ECE", "AUROC": "AUROC",
    "Coverage_90": "coverage90", "Traj_RMSE_continuation": "traj_rmse_continuation",
    "N_liq_logMAE": "nliq_log_err",
}


def headline_table(summary_df: pd.DataFrame, cluster_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Headline table in P2 metric order. Point estimates from ``summary_df`` (*_mean); confidence
    intervals preferentially from the **object-cluster bootstrap** ``cluster_df`` (correct for grouped
    CV). Metrics absent from the bootstrap show the point estimate only (no naive fold CI is shown)."""
    cl = cluster_df.set_index("model") if cluster_df is not None else None
    rows = []
    for _, r in summary_df.iterrows():
        model = r["model"]; rec = {"model": model}
        for key, label in HEADLINE_ORDER:
            mcol = f"{key}_mean"
            if mcol not in summary_df.columns or pd.isna(r[mcol]):
                rec[label] = "—"; continue
            v = r[mcol]; ck = _CLUSTER_CI_MAP.get(key)
            if cl is not None and ck is not None and model in cl.index and f"{ck}_lo" in cl.columns \
                    and pd.notna(cl.loc[model, f"{ck}_lo"]):
                # ТОЧКА и CI — из ОДНОГО оценщика (object-cluster bootstrap), иначе точка может
                # оказаться вне интервала (как Transformer 0.050 [0.022, 0.047]). Берём bootstrap-точку.
                pt = cl.loc[model, ck] if ck in cl.columns and pd.notna(cl.loc[model, ck]) else v
                rec[label] = f"{pt:.3f} [{cl.loc[model, f'{ck}_lo']:.3f}, {cl.loc[model, f'{ck}_hi']:.3f}]"
            else:
                rec[label] = f"{v:.3f}"
        rows.append(rec)
    return pd.DataFrame(rows)
