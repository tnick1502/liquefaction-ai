"""
Фабрика объединённых публикационных панелей (matplotlib, Q1).

Собирает несколько связанных графиков в одну многопанельную фигуру — то, что идёт в статью,
где нельзя вставить все рисунки по отдельности. Три ключевые панели:

* :func:`data_overview_panel` — обзор данных (типы грунта, PLAXIS-классы, физмех-распределения,
  примеры кривых PPR(N));
* :func:`crr_physics_panel` — физика сопротивления разжижению CRR (кривые CRR(N), зависимости
  от свойств грунта, разложение α/β);
* :func:`model_leaderboard_panel` — сравнение моделей (P³-скор, траекторная ошибка,
  AUROC↔Brier, ошибка N_liq).

Все панели возвращают :class:`liquefaction_ai.viz.figure_io.MplFig` и сохраняются в высоком
разрешении PNG/PDF через :func:`liquefaction_ai.viz.save_figure`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from liquefaction_ai.viz.figure_io import MplFig, new_figure, save_figure
from liquefaction_ai.viz.theme import GRID, INK, QUALITATIVE, SEQUENTIAL, plain_log_axis

__all__ = ["data_overview_panel", "crr_physics_panel", "model_leaderboard_panel"]


def _panel_label(ax, letter: str) -> None:
    """Подпись панели (a), (b)… в журнальном стиле."""
    ax.text(-0.08, 1.06, f"({letter})", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right", color=INK)


def _clean(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def data_overview_panel(
    meta,
    cycles: np.ndarray,
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    soil_label_map: Optional[Dict[str, str]] = None,
    title: str = "Dataset overview",
    save: bool = False,
    fig_id: str = "5_1_data_overview",
) -> MplFig:
    """
    Панель обзора данных (2×2): состав по типам грунта, PLAXIS-классы, e–V_s по разжижению,
    примеры гладких кривых PPR(N).

    :param meta: таблица метаданных популяции (нужны ``soil_type``/``plaxis_class``/``e``/
                 ``V_s``/``liq_label``/``CSR_base``)
    :param cycles: сетка циклов (n, T)
    :param r_obs: измеренная PPR(N) (n, T)
    :param valid_mask: маска валидности (n, T)
    :param soil_label_map: подписи типов грунта (``soil_type`` → текст)
    :param title: общий заголовок
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    figw, fig = new_figure((11.0, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)

    # (a) состав по типам грунта
    ax = fig.add_subplot(gs[0, 0])
    vc = meta["soil_type"].value_counts()
    labels = [(soil_label_map or {}).get(s, s) for s in vc.index]
    ax.barh(range(len(vc)), vc.to_numpy(), color=QUALITATIVE[0], edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(vc))); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    for i, v in enumerate(vc.to_numpy()):
        ax.text(v, i, f" {v}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("number of tests"); ax.set_title("Soil-type composition"); _clean(ax); _panel_label(ax, "a")

    # (b) PLAXIS-классы
    ax = fig.add_subplot(gs[0, 1])
    if "plaxis_class" in meta.columns:
        order = ["very fine", "fine", "medium", "coarse", "very coarse"]
        pc = meta["plaxis_class"].value_counts()
        cats = [c for c in order if c in pc.index] + [c for c in pc.index if c not in order]
        vals = [int(pc.get(c, 0)) for c in cats]
        ax.bar(range(len(cats)), vals, color=QUALITATIVE[2], edgecolor="white", linewidth=0.5, width=0.66)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylabel("number of tests"); ax.set_title("PLAXIS grain-size classes"); _clean(ax); _panel_label(ax, "b")

    # (c) e vs V_s, раскраска по разжижению
    ax = fig.add_subplot(gs[1, 0])
    liq = meta["liq_label"].to_numpy().astype(bool)
    ax.scatter(meta["e"].to_numpy()[~liq], meta["V_s"].to_numpy()[~liq], s=22, alpha=0.7,
               color=QUALITATIVE[7], edgecolors="white", linewidths=0.25, label="stable")
    ax.scatter(meta["e"].to_numpy()[liq], meta["V_s"].to_numpy()[liq], s=22, alpha=0.7,
               color=QUALITATIVE[1], edgecolors="white", linewidths=0.25, label="liquefied")
    ax.set_xlabel("void ratio e (–)"); ax.set_ylabel("shear-wave velocity Vs (m/s)")
    ax.set_title("Void ratio vs Vs by outcome"); ax.legend(fontsize=8.5); _clean(ax); _panel_label(ax, "c")

    # (d) примеры PPR(N)
    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(meta), size=min(14, len(meta)), replace=False)
    csr = meta["CSR_base"].to_numpy()
    norm = (csr - csr.min()) / (np.ptp(csr) + 1e-9)
    for i in idx:
        m = valid_mask[i] > 0
        ax.plot(cycles[i][m], r_obs[i][m], color=SEQUENTIAL(float(norm[i])), linewidth=1.4, alpha=0.85)
    ax.axhline(1.0, ls="--", color="#c46b6b", linewidth=1.2)
    ax.set_xlabel("number of cycles N"); ax.set_ylabel("PPR (–)")
    ax.set_title("Measured pore-pressure curves PPR(N)"); _clean(ax); _panel_label(ax, "d")

    fig.suptitle(title, y=0.99, fontsize=15)
    return save_figure(figw, fig_id, save)


def crr_physics_panel(
    meta,
    title: str = "Cyclic resistance (CRR) physics",
    save: bool = False,
    fig_id: str = "5_1_crr_physics",
) -> MplFig:
    """
    Панель физики CRR (2×2): кривые CRR(N)=β/N^(1−α) по типам грунта, CRR15 vs D_r, CRR15 vs I_p,
    разложение α/β.

    :param meta: таблица метаданных (нужны ``crr_alpha``/``crr_betta``/``crr_ref``/``D_r``/
                 ``I_p``/``soil_type``/``class_id``)
    :param title: общий заголовок
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    figw, fig = new_figure((11.0, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)
    N = np.linspace(1, 100, 200)

    # (a) кривые CRR(N) по типам грунта (медианные α/β)
    ax = fig.add_subplot(gs[0, 0])
    grp = meta.groupby("soil_type")[["crr_alpha", "crr_betta"]].median()
    for i, (st, row) in enumerate(grp.iterrows()):
        crr = row["crr_betta"] / N ** (1.0 - row["crr_alpha"])
        ax.plot(N, crr, color=QUALITATIVE[i % len(QUALITATIVE)], linewidth=2.0, label=st)
    ax.set_xscale("log"); plain_log_axis(ax, "x"); ax.set_xlabel("number of cycles N"); ax.set_ylabel("CRR (–)")
    ax.set_title("CRR(N) = β / N^(1−α) by soil type"); ax.legend(fontsize=7.5, ncol=2); _clean(ax); _panel_label(ax, "a")

    # (b) CRR15 vs относительная плотность
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(meta["D_r"].to_numpy(), meta["crr_ref"].to_numpy(), s=20, alpha=0.7,
                    c=meta["I_p"].to_numpy(), cmap=SEQUENTIAL, edgecolors="white", linewidths=0.2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("plasticity Ip")
    ax.set_xlabel("relative density D_r (–)"); ax.set_ylabel("CRR15 (–)")
    ax.set_title("Cyclic resistance vs density"); _clean(ax); _panel_label(ax, "b")

    # (c) CRR15 vs число пластичности
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(meta["I_p"].to_numpy(), meta["crr_ref"].to_numpy(), s=20, alpha=0.65,
               color=QUALITATIVE[4], edgecolors="white", linewidths=0.2)
    ax.set_xlabel("plasticity index Ip (–)"); ax.set_ylabel("CRR15 (–)")
    ax.set_title("Cyclic resistance vs plasticity"); _clean(ax); _panel_label(ax, "c")

    # (d) разложение α/β по типам грунта
    ax = fig.add_subplot(gs[1, 1])
    st = list(grp.index); x = np.arange(len(st)); w = 0.38
    ax.bar(x - w / 2, grp["crr_alpha"].to_numpy(), width=w, color=QUALITATIVE[0],
           edgecolor="white", linewidth=0.4, label="α (rate)")
    ax.bar(x + w / 2, grp["crr_betta"].to_numpy(), width=w, color=QUALITATIVE[3],
           edgecolor="white", linewidth=0.4, label="β (amplitude)")
    ax.set_xticks(x); ax.set_xticklabels(st, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("parameter value"); ax.set_title("CRR-curve parameters α, β"); ax.legend(fontsize=8.5)
    _clean(ax); _panel_label(ax, "d")

    fig.suptitle(title, y=0.99, fontsize=15)
    return save_figure(figw, fig_id, save)


def model_leaderboard_panel(
    leaderboard,
    score_col: str = "P3_Core_Raw_Score",
    highlight: Sequence[str] = ("DPI-EVT", "EVT-NeuralSSM", "DPI-Flow"),
    title: str = "Model comparison",
    save: bool = False,
    fig_id: str = "5_1_leaderboard",
) -> MplFig:
    """
    Панель сравнения моделей (2×2): P³-скор, траекторная ошибка, AUROC↔Brier, ошибка N_liq.

    Использует колонки таблицы-лидерборда (``model``, ``score_col``, ``Traj_RMSE``, ``AUROC``,
    ``Brier``, ``N_liq_logMAE``/``N_liq_MAE``). Недостающие колонки пропускаются.

    :param leaderboard: таблица метрик по моделям (DataFrame с колонкой ``model``)
    :param score_col: колонка главного скора (по умолчанию ``P3_Core_Raw_Score``)
    :param highlight: имена флагманских моделей для выделения цветом
    :param title: общий заголовок
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    df = leaderboard.copy()
    figw, fig = new_figure((11.0, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    def _hl(names):
        return [QUALITATIVE[2] if str(n) in highlight else QUALITATIVE[7] for n in names]

    # (a) P3-скор
    ax = fig.add_subplot(gs[0, 0])
    if score_col in df.columns:
        d = df.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
        ax.barh(range(len(d)), d[score_col].to_numpy(), color=_hl(d["model"]), edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["model"], fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("P³-Core score ↑")
    ax.set_title("Publication P³ score"); _clean(ax); _panel_label(ax, "a")

    # (b) траекторная ошибка
    ax = fig.add_subplot(gs[0, 1])
    if "Traj_RMSE" in df.columns:
        d = df.dropna(subset=["Traj_RMSE"]).sort_values("Traj_RMSE")
        ax.barh(range(len(d)), d["Traj_RMSE"].to_numpy(), color=_hl(d["model"]), edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["model"], fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("trajectory RMSE ↓")
    ax.set_title("PPR(N) trajectory error"); _clean(ax); _panel_label(ax, "b")

    # (c) AUROC ↔ Brier
    ax = fig.add_subplot(gs[1, 0])
    if {"AUROC", "Brier"}.issubset(df.columns):
        d = df.dropna(subset=["AUROC", "Brier"])
        for _, r in d.iterrows():
            hl = str(r["model"]) in highlight
            ax.scatter(r["Brier"], r["AUROC"], s=70 if hl else 38,
                       color=QUALITATIVE[2] if hl else QUALITATIVE[7],
                       edgecolors=INK, linewidths=0.5, zorder=3 if hl else 2)
            if hl:
                ax.annotate(str(r["model"]), (r["Brier"], r["AUROC"]), fontsize=7.5,
                            xytext=(4, 3), textcoords="offset points", color=INK)
        ax.set_xlabel("Brier ↓"); ax.set_ylabel("AUROC ↑")
    ax.set_title("Risk classification quality"); _clean(ax); _panel_label(ax, "c")

    # (d) ошибка N_liq
    ax = fig.add_subplot(gs[1, 1])
    col = "N_liq_logMAE" if "N_liq_logMAE" in df.columns else ("N_liq_MAE" if "N_liq_MAE" in df.columns else None)
    if col:
        d = df.dropna(subset=[col]).sort_values(col)
        ax.barh(range(len(d)), d[col].to_numpy(), color=_hl(d["model"]), edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["model"], fontsize=8); ax.invert_yaxis()
        ax.set_xlabel(f"{col} ↓")
    ax.set_title("Cycles-to-liquefaction error"); _clean(ax); _panel_label(ax, "d")

    fig.suptitle(title, y=0.99, fontsize=15)
    return save_figure(figw, fig_id, save)


def admissible_pareto_panel(
    leaderboard,
    x_col: str = "Physics_Violation_Rate",
    y_col: str = "N_liq_logMAE",
    highlight: str = "DPI-Flow",
    title: str = "Admissible onset Pareto: physics violations vs onset error",
    save: bool = False,
    fig_id: str = "3_7_admissible_pareto",
) -> MplFig:
    """
    Pareto-фигура «допустимость ↔ onset»: по X — доля физических нарушений (↓), по Y — ошибка
    censored N_liq (↓). Левый-нижний угол = физически допустимые И точные по onset модели.

    Показывает центральный claim #1: DPI-Flow — на admissible-фронте (нулевые/около-нулевые
    нарушения + лучший onset). Модели с лучшим RMSE, но высокими нарушениями (RealNVP/Transformer)
    оказываются справа и недопустимы.

    :param leaderboard: таблица метрик по моделям (DataFrame с колонкой ``model``)
    :param x_col: колонка оси X (доля нарушений)
    :param y_col: колонка оси Y (ошибка onset)
    :param highlight: модель для выделения
    :param title: заголовок
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    df = leaderboard.dropna(subset=[x_col, y_col]).copy()
    figw, fig = new_figure((7.2, 5.4))
    ax = fig.add_subplot(111)
    xs = df[x_col].to_numpy(); ys = df[y_col].to_numpy(); names = df["model"].astype(str).to_numpy()

    # Pareto-фронт (минимизация обеих осей)
    order = np.argsort(xs)
    front, best_y = [], np.inf
    for i in order:
        if ys[i] <= best_y + 1e-12:
            front.append(i); best_y = ys[i]
    front = sorted(front, key=lambda i: xs[i])
    ax.plot(xs[front], ys[front], "--", color=GRID, linewidth=1.3, zorder=1, label="admissible Pareto front")

    for x, y, nm in zip(xs, ys, names):
        is_hl = nm == highlight
        ax.scatter(x, y, s=120 if is_hl else 55,
                   color=QUALITATIVE[2] if is_hl else QUALITATIVE[7],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(nm, (x, y), fontsize=8.5 if is_hl else 7.5,
                    xytext=(5, 4), textcoords="offset points",
                    fontweight="bold" if is_hl else "normal", color=INK)
    ax.set_xlabel(f"{x_col.replace('_', ' ')} ↓ (inadmissible →)")
    ax.set_ylabel(f"{y_col.replace('_', ' ')} ↓ (worse onset ↑)")
    ax.set_title(title); ax.legend(fontsize=8.5, loc="upper right"); _clean(ax)
    return save_figure(figw, fig_id, save)
