"""
Библиотека построителей графиков matplotlib (журнальный стиль Q1) для ноутбуков проекта.

Каждый построитель создаёт оформленную в едином пастельном стиле фигуру и принимает флаг
``save``: при ``save=True`` фигура сохраняется в ``results/figs/`` в высоком разрешении PNG
(и PDF) под именем ``fig_id`` (см. :func:`liquefaction_ai.viz.figure_io.save_figure`).
Набор покрывает распределения, box-плоты по группам, корреляционные карты, диаграммы
рассеяния, трёхмерные поверхности, столбчатые диаграммы, линейные графики, полосы
неопределённости, калибровочные кривые и дашборды обучения/grid-search.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import to_rgba
from scipy.interpolate import griddata

from liquefaction_ai.viz.figure_io import MplFig, new_figure, save_figure
from liquefaction_ai.viz.theme import DIVERGING, GRID, INK, QUALITATIVE, SEQUENTIAL

__all__ = [
    "histogram_grid",
    "box_grid",
    "correlation_heatmap",
    "scatter",
    "scatter_by_group",
    "surface3d_grid",
    "bar",
    "grouped_bar",
    "lines",
    "line_with_bands",
    "calibration_plot",
    "training_dashboard",
    "grid_search_plot",
    "grid_search_dashboard",
    "heatmap",
]

_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-.", "longdash": (0, (6, 3))}
_BEST = "#7FB286"   # пастельно-зелёный — лучшая конфигурация
_BASE = "#5B8FB0"   # пыльно-синий — базовый


def _resolve_cmap(cmap):
    """Привести цветовую карту (объект или имя, в т.ч. plotly-стиля) к matplotlib cmap."""
    import matplotlib as mpl
    from matplotlib.colors import Colormap
    if cmap is None or isinstance(cmap, Colormap):
        return cmap if cmap is not None else SEQUENTIAL
    if isinstance(cmap, str):
        for name in (cmap, cmap.lower()):
            try:
                return mpl.colormaps[name]
            except (KeyError, AttributeError):
                continue
    return SEQUENTIAL


def _color(c, default: str = QUALITATIVE[0]):
    """Привести цвет (hex/имя/``rgba(...)``) к виду, понятному matplotlib."""
    if c is None:
        return default
    if isinstance(c, str) and c.startswith("rgba(") and c.endswith(")"):
        parts = c[5:-1].split(",")
        r, g, b = (float(parts[0]) / 255, float(parts[1]) / 255, float(parts[2]) / 255)
        a = float(parts[3]) if len(parts) > 3 else 1.0
        return (r, g, b, a)
    return c


def _style_axis(ax) -> None:
    """Единое оформление оси: мягкая горизонтальная сетка, чистые рамки."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend(ax, **kw) -> None:
    """Аккуратная легенда в едином стиле."""
    leg = ax.legend(frameon=True, framealpha=0.85, edgecolor=GRID, fancybox=False, **kw)
    if leg:
        leg.get_frame().set_linewidth(0.7)


def grid_search_dashboard(
    results,
    metric_keys: Sequence[str],
    param_cols: Sequence[str],
    selected_metric: str,
    metric_labels: Optional[Dict[str, str]] = None,
    metric_fmts: Optional[Dict[str, str]] = None,
    lower_is_better: bool = True,
    target: Optional[float] = None,
    title: str = "Grid search — metrics by configuration",
    row_height: int = 210,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Мультипанельный дашборд grid search: по Y — метрики, по X — конфигурации гиперпараметров.

    :param results: таблица результатов (DataFrame со столбцами параметров и метрик)
    :param metric_keys: ключи метрик (по панели на метрику)
    :param param_cols: столбцы гиперпараметров для текстовой подписи конфигураций
    :param selected_metric: метрика, по которой выбрана лучшая конфигурация (подсветка)
    :param metric_labels: подписи панелей «ключ → название»
    :param metric_fmts: форматы чисел по метрикам
    :param lower_is_better: направление ``selected_metric``
    :param target: целевое значение ``selected_metric`` (лучший — по близости)
    :param title: общий заголовок
    :param row_height: высота одной панели, пикс.
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    df = results.reset_index(drop=True)
    metric_labels = metric_labels or {}
    metric_fmts = metric_fmts or {}
    labels = [", ".join(f"{p}={df.loc[i, p]}" for p in param_cols) for i in range(len(df))]
    if target is not None:
        best = int((df[selected_metric] - target).abs().idxmin())
    else:
        best = int(df[selected_metric].idxmin() if lower_is_better else df[selected_metric].idxmax())

    n = len(metric_keys)
    figw, fig = new_figure((max(7.0, 0.9 * len(df) + 2.5), row_height / 96 * n))
    axes = fig.subplots(n, 1, sharex=True, squeeze=False)[:, 0]
    x = np.arange(len(df))
    for ax, key in zip(axes, metric_keys):
        colors = [_BEST if i == best else _BASE for i in range(len(df))]
        ax.bar(x, df[key].to_numpy(dtype=float), color=colors, width=0.66, edgecolor="white", linewidth=0.5)
        fmt = metric_fmts.get(key, ".4f")
        for xi, v in zip(x, df[key].to_numpy(dtype=float)):
            if np.isfinite(v):
                ax.text(xi, v, format(v, fmt), ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_ylabel(metric_labels.get(key, key), fontsize=9.5)
        _style_axis(ax)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[-1].set_xlabel("Hyper-parameter configuration")
    fig.suptitle(f"{title}\n(best by {selected_metric} highlighted)", y=1.0)
    fig.tight_layout()
    return save_figure(figw, fig_id, save)


def heatmap(
    matrix,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str = "",
    colorscale=SEQUENTIAL,
    value_fmt: Optional[str] = ".2f",
    colorbar_title: str = "",
    height: int = 520,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Аннотированная тепловая карта произвольной матрицы (не обязательно квадратной).

    :param matrix: двумерный массив, форма (len(y_labels), len(x_labels))
    :param x_labels: подписи столбцов
    :param y_labels: подписи строк
    :param title: заголовок
    :param colorscale: matplotlib colormap (по умолчанию пастельная последовательная)
    :param value_fmt: формат аннотаций (или None — без аннотаций)
    :param colorbar_title: подпись цветовой шкалы
    :param height: высота фигуры, пикс.
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    z = np.asarray(matrix, dtype=float)
    x_labels, y_labels = list(x_labels), list(y_labels)
    figw, fig = new_figure((max(6.0, 0.7 * len(x_labels) + 2), height / 96))
    ax = fig.add_subplot(111)
    im = ax.imshow(z, cmap=_resolve_cmap(colorscale), aspect="auto")
    ax.set_xticks(range(len(x_labels))); ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels))); ax.set_yticklabels(y_labels)
    if value_fmt:
        lo, hi = np.nanmin(z), np.nanmax(z); mid = (lo + hi) / 2
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                ax.text(j, i, format(z[i, j], value_fmt), ha="center", va="center",
                        fontsize=7.5, color="white" if z[i, j] > mid else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_title)
    ax.set_title(title); ax.grid(False)
    fig.tight_layout()
    return save_figure(figw, fig_id, save)


def histogram_grid(
    df,
    columns: Sequence[str],
    titles: Optional[Sequence[str]] = None,
    n_cols: int = 3,
    colors: Optional[Sequence[str]] = None,
    nbins: int = 40,
    title: str = "Распределения параметров",
    row_height: int = 300,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Сетка гистограмм распределений выбранных колонок таблицы.

    :param df: таблица данных
    :param columns: колонки для гистограмм
    :param titles: заголовки подграфиков (по умолчанию имена колонок)
    :param n_cols: число столбцов сетки
    :param colors: цвета гистограмм
    :param nbins: число интервалов
    :param title: общий заголовок
    :param row_height: высота ряда, пикс.
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    titles = list(titles) if titles is not None else list(columns)
    colors = list(colors) if colors is not None else QUALITATIVE
    n_rows = int(np.ceil(len(columns) / n_cols))
    figw, fig = new_figure((4.6 * n_cols, row_height / 96 * n_rows))
    axes = np.atleast_1d(fig.subplots(n_rows, n_cols, squeeze=False)).ravel()
    for i, col in enumerate(columns):
        ax = axes[i]
        vals = np.asarray(df[col], dtype=float); vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=nbins, color=colors[i % len(colors)], alpha=0.88,
                edgecolor="white", linewidth=0.3)
        ax.set_title(titles[i], fontsize=11)
        _style_axis(ax)
    for k in range(len(columns), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return save_figure(figw, fig_id, save)


def box_grid(
    df,
    value_cols: Sequence[str],
    group_col: str,
    group_order: Sequence[str],
    group_labels: Optional[Dict[str, str]] = None,
    titles: Optional[Sequence[str]] = None,
    n_cols: int = 3,
    colors: Optional[Sequence[str]] = None,
    title: str = "Распределения по группам",
    row_height: int = 330,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Сетка box-плотов: распределение каждой величины в разрезе групп.

    :param df: таблица данных
    :param value_cols: числовые колонки (по подграфику на колонку)
    :param group_col: колонка группировки
    :param group_order: порядок групп
    :param group_labels: подписи групп
    :param titles: заголовки подграфиков
    :param n_cols: число столбцов сетки
    :param colors: цвета групп
    :param title: общий заголовок
    :param row_height: высота ряда, пикс.
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    titles = list(titles) if titles is not None else list(value_cols)
    colors = list(colors) if colors is not None else QUALITATIVE
    labels = [(group_labels or {}).get(g, g) for g in group_order]
    n_rows = int(np.ceil(len(value_cols) / n_cols))
    figw, fig = new_figure((4.8 * n_cols, row_height / 96 * n_rows))
    axes = np.atleast_1d(fig.subplots(n_rows, n_cols, squeeze=False)).ravel()
    for i, col in enumerate(value_cols):
        ax = axes[i]
        data = [df.loc[df[group_col] == g, col].to_numpy(dtype=float) for g in group_order]
        data = [d[np.isfinite(d)] for d in data]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.62,
                        medianprops=dict(color=INK, linewidth=1.3),
                        meanprops=dict(marker="o", markerfacecolor="white",
                                       markeredgecolor=INK, markersize=4),
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.4,
                                        markerfacecolor=GRID, markeredgecolor="none"),
                        whiskerprops=dict(color="#9aa3ad"), capprops=dict(color="#9aa3ad"))
        for patch, gi in zip(bp["boxes"], range(len(group_order))):
            patch.set_facecolor(to_rgba(colors[gi % len(colors)], 0.78))
            patch.set_edgecolor("white"); patch.set_linewidth(0.8)
        ax.set_title(titles[i], fontsize=11)
        ax.set_xticks(range(1, len(group_order) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        _style_axis(ax)
    for k in range(len(value_cols), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return save_figure(figw, fig_id, save)


def correlation_heatmap(
    corr,
    title: str = "Корреляционная матрица",
    save: bool = False,
    fig_id: str = "",
    height: int = 720,
) -> MplFig:
    """
    Тепловая карта корреляционной матрицы с числовыми аннотациями (пастельная diverging).

    :param corr: квадратная таблица корреляций (DataFrame)
    :param title: заголовок
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :param height: высота фигуры, пикс.
    :return: обёртка фигуры :class:`MplFig`
    """
    labels = list(corr.columns)
    z = np.asarray(corr.to_numpy(), dtype=float)
    side = max(5.5, height / 110)
    figw, fig = new_figure((side, side))
    ax = fig.add_subplot(111)
    im = ax.imshow(z, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(z[i, j]) > 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("r")
    ax.set_title(title); ax.grid(False)
    fig.tight_layout()
    return save_figure(figw, fig_id, save)


def scatter(
    x, y,
    color=None,
    color_label: str = "",
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    logx: bool = False,
    logy: bool = False,
    hline: Optional[float] = None,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Диаграмма рассеяния с опциональной непрерывной окраской точек.

    :param x: значения по оси X
    :param y: значения по оси Y
    :param color: значения непрерывной окраски (или None)
    :param color_label: подпись цветовой шкалы
    :param title: заголовок
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param logx: лог-шкала X
    :param logy: лог-шкала Y
    :param hline: уровень горизонтальной опорной линии
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    figw, fig = new_figure((7.0, 4.6))
    ax = fig.add_subplot(111)
    if color is not None:
        sc = ax.scatter(np.asarray(x), np.asarray(y), c=np.asarray(color), cmap=SEQUENTIAL,
                        s=22, alpha=0.8, edgecolors="white", linewidths=0.25)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label(color_label)
    else:
        ax.scatter(np.asarray(x), np.asarray(y), s=22, alpha=0.65, color=QUALITATIVE[0],
                   edgecolors="white", linewidths=0.25)
    if hline is not None:
        ax.axhline(hline, ls="--", color="#c46b6b", linewidth=1.3)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def scatter_by_group(
    df,
    x: str,
    y: str,
    group_col: str,
    group_order: Sequence[str],
    group_labels: Optional[Dict[str, str]] = None,
    colors: Optional[Sequence[str]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Диаграмма рассеяния, раскрашенная по категориальной группе.

    :param df: таблица данных
    :param x: колонка оси X
    :param y: колонка оси Y
    :param group_col: колонка группы
    :param group_order: порядок групп
    :param group_labels: подписи групп
    :param colors: цвета групп
    :param title: заголовок
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    colors = list(colors) if colors is not None else QUALITATIVE
    figw, fig = new_figure((7.0, 4.6))
    ax = fig.add_subplot(111)
    for gi, g in enumerate(group_order):
        sub = df[df[group_col] == g]
        ax.scatter(sub[x], sub[y], s=20, alpha=0.6, color=colors[gi % len(colors)],
                   edgecolors="white", linewidths=0.2,
                   label=(group_labels or {}).get(g, g))
    ax.set_title(title); ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y)
    _legend(ax, fontsize=8); _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def _interp_grid(x, y, z, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Интерполировать рассеянные точки на регулярную сетку (линейно + ближайший сосед)."""
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    xg = np.linspace(x.min(), x.max(), resolution)
    yg = np.linspace(y.min(), y.max(), resolution)
    xx, yy = np.meshgrid(xg, yg)
    zz = griddata((x, y), z, (xx, yy), method="linear")
    if np.isnan(zz).any():
        near = griddata((x, y), z, (xx, yy), method="nearest")
        zz = np.where(np.isnan(zz), near, zz)
    return xx, yy, zz


def surface3d_grid(
    specs: Sequence[Dict],
    n_cols: int = 3,
    resolution: int = 40,
    title: str = "Трёхмерные поверхности",
    height: int = 520,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Сетка интерполированных трёхмерных поверхностей.

    :param specs: список спецификаций; ключи ``x``/``y``/``z`` (данные), ``title``,
                  ``xlabel``/``ylabel``/``zlabel`` и опционально ``colorscale``
    :param n_cols: число поверхностей в ряду
    :param resolution: разрешение интерполяционной сетки
    :param title: общий заголовок
    :param height: высота ряда, пикс.
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    n = len(specs); n_rows = int(np.ceil(n / n_cols))
    figw, fig = new_figure((4.8 * n_cols, height / 96 * n_rows))
    for i, s in enumerate(specs):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        xx, yy, zz = _interp_grid(s["x"], s["y"], s["z"], resolution)
        ax.plot_surface(xx, yy, zz, cmap=_resolve_cmap(s.get("colorscale")), linewidth=0,
                        antialiased=True, alpha=0.95)
        ax.set_title(s.get("title", ""), fontsize=10.5)
        ax.set_xlabel(s.get("xlabel", "x"), fontsize=8); ax.set_ylabel(s.get("ylabel", "y"), fontsize=8)
        ax.set_zlabel(s.get("zlabel", "z"), fontsize=8)
        ax.view_init(elev=24, azim=-130)
        ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04); ax.zaxis.pane.set_alpha(0.04)
    fig.suptitle(title, y=1.0); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def bar(
    categories: Sequence[str],
    values: Sequence[float],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = QUALITATIVE[0],
    horizontal: bool = False,
    text_fmt: Optional[str] = ".3f",
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Столбчатая диаграмма (вертикальная или горизонтальная).

    :param categories: подписи категорий
    :param values: значения столбцов
    :param title: заголовок
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param color: цвет столбцов
    :param horizontal: горизонтальная ориентация
    :param text_fmt: формат подписи значений (или None)
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    cats = list(categories); values = np.asarray(values, dtype=float)
    figw, fig = new_figure((max(6.4, 0.5 * len(cats) + 2.5), 4.4))
    ax = fig.add_subplot(111)
    if horizontal:
        ax.barh(cats, values, color=color, edgecolor="white", linewidth=0.5)
        ax.invert_yaxis()
        if text_fmt:
            for i, v in enumerate(values):
                if np.isfinite(v): ax.text(v, i, " " + format(v, text_fmt), va="center", fontsize=8, color=INK)
    else:
        xpos = np.arange(len(cats))
        ax.bar(xpos, values, color=color, edgecolor="white", linewidth=0.5, width=0.66)
        ax.set_xticks(xpos); ax.set_xticklabels(cats, rotation=25, ha="right")
        if text_fmt:
            for i, v in enumerate(values):
                if np.isfinite(v): ax.text(i, v, format(v, text_fmt), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def grouped_bar(
    categories: Sequence[str],
    series: Dict[str, Sequence[float]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: Optional[Sequence[str]] = None,
    barmode: str = "group",
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Сгруппированная (или составная) столбчатая диаграмма для сравнения рядов.

    :param categories: подписи категорий по X
    :param series: словарь «имя ряда → значения»
    :param title: заголовок
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param colors: цвета рядов
    :param barmode: ``group`` или ``stack``
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    colors = list(colors) if colors is not None else QUALITATIVE
    cats = list(categories); names = list(series.keys()); m = len(names)
    x = np.arange(len(cats))
    figw, fig = new_figure((max(6.6, 0.7 * len(cats) + 2.5), 4.4))
    ax = fig.add_subplot(111)
    if barmode == "stack":
        bottom = np.zeros(len(cats))
        for i, nm in enumerate(names):
            vals = np.asarray(series[nm], dtype=float)
            ax.bar(x, vals, bottom=bottom, color=colors[i % len(colors)], label=nm,
                   edgecolor="white", linewidth=0.4)
            bottom += np.nan_to_num(vals)
    else:
        w = 0.8 / max(m, 1)
        for i, nm in enumerate(names):
            ax.bar(x + (i - (m - 1) / 2) * w, np.asarray(series[nm], dtype=float), width=w,
                   color=colors[i % len(colors)], label=nm, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _legend(ax, fontsize=8); _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def lines(
    series: Sequence[Dict],
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    logx: bool = False,
    logy: bool = False,
    hline: Optional[float] = None,
    height: int = 480,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Линейный график из нескольких рядов (кривые обучения, траектории и т.п.).

    :param series: список рядов; каждый — словарь с ключами ``x``, ``y``, ``name`` и
                   опционально ``mode``/``color``/``dash``/``width``
    :param title: заголовок
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param logx: лог-шкала X
    :param logy: лог-шкала Y
    :param hline: уровень горизонтальной опорной линии
    :param height: высота фигуры, пикс.
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    figw, fig = new_figure((7.4, height / 96))
    ax = fig.add_subplot(111)
    show_leg = False
    for i, s in enumerate(series):
        mode = s.get("mode", "lines")
        marker = "o" if "markers" in mode else None
        ls = "none" if mode == "markers" else _DASH.get(s.get("dash", "solid"), "-")
        nm = s.get("name")
        if nm: show_leg = True
        ax.plot(np.asarray(s["x"]), np.asarray(s["y"]), label=nm, marker=marker,
                linestyle=ls, color=_color(s.get("color"), QUALITATIVE[i % len(QUALITATIVE)]),
                linewidth=s.get("width", 2.2), markersize=4.5, markeredgecolor="white", markeredgewidth=0.3)
    if hline is not None:
        ax.axhline(hline, ls="--", color="#c46b6b", linewidth=1.3)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if show_leg: _legend(ax, fontsize=8.5)
    _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def line_with_bands(
    x,
    mean,
    lower,
    upper,
    extra: Optional[Sequence[Dict]] = None,
    band_color="#5B8FB0",
    line_color: str = "#3c5f86",
    mean_name: str = "Среднее",
    band_name: str = "90% интервал",
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    logx: bool = False,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Линия со средним и закрашенной полосой неопределённости.

    :param x: значения по оси X
    :param mean: среднее предсказание
    :param lower: нижняя граница интервала
    :param upper: верхняя граница интервала
    :param extra: дополнительные ряды (словари ``x``/``y``/``name``/``color``/``dash``)
    :param band_color: цвет заливки полосы
    :param line_color: цвет линии среднего
    :param mean_name: подпись линии среднего
    :param band_name: подпись полосы
    :param title: заголовок
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param logx: лог-шкала X
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    x = np.asarray(x)
    figw, fig = new_figure((7.4, 4.6))
    ax = fig.add_subplot(111)
    ax.fill_between(x, np.asarray(lower), np.asarray(upper), color=to_rgba(_color(band_color), 0.22),
                    linewidth=0, label=band_name)
    ax.plot(x, np.asarray(mean), color=_color(line_color), linewidth=2.4, label=mean_name)
    for s in (extra or []):
        ax.plot(np.asarray(s["x"]), np.asarray(s["y"]), label=s.get("name", ""),
                color=_color(s.get("color"), INK), linestyle=_DASH.get(s.get("dash", "solid"), "-"),
                linewidth=s.get("width", 2.0))
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if logx: ax.set_xscale("log")
    _legend(ax, fontsize=8.5); _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def training_dashboard(
    history,
    title: str = "Training dynamics",
    model_color: str = "#5B8FB0",
    save: bool = False,
    fig_id: str = "",
    height: int = 420,
) -> MplFig:
    """
    Дашборд динамики обучения: потери и валидационные метрики по эпохам.

    :param history: история обучения (DataFrame: ``epoch``, ``train_loss``, ``val_loss`` и
                    опционально ``val_auroc``/``val_traj_rmse``)
    :param title: общий заголовок
    :param model_color: базовый цвет модели
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :param height: высота фигуры, пикс.
    :return: обёртка фигуры :class:`MplFig`
    """
    epoch = np.asarray(history["epoch"])
    panels = [("loss", "Loss (–)")]
    if "val_auroc" in history: panels.append(("val_auroc", "Validation AUROC (–)"))
    if "val_traj_rmse" in history: panels.append(("val_traj_rmse", "Validation trajectory RMSE (–)"))
    n = len(panels)
    figw, fig = new_figure((4.8 * n, height / 96))
    axes = np.atleast_1d(fig.subplots(1, n, squeeze=False)).ravel()
    for ax, (key, ylab) in zip(axes, panels):
        if key == "loss":
            ax.plot(epoch, history["train_loss"], "-o", color=_color(model_color), label="train loss", markersize=4)
            ax.plot(epoch, history["val_loss"], "--s", color="#CB8FB3", label="val loss", markersize=4)
            _legend(ax, fontsize=8)
        else:
            ax.plot(epoch, history[key], "-o", color="#7FB286", markersize=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylab); _style_axis(ax)
    fig.suptitle(title, y=1.02); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def grid_search_plot(
    results,
    score_col: str,
    param_cols: Sequence[str],
    title: str = "Grid search results",
    xlabel: Optional[str] = None,
    lower_is_better: bool = True,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Горизонтальная диаграмма результатов перебора гиперпараметров (лучший — выделен).

    :param results: таблица результатов (DataFrame)
    :param score_col: колонка метрики отбора
    :param param_cols: колонки гиперпараметров для подписи комбинаций
    :param title: заголовок
    :param xlabel: подпись оси значений
    :param lower_is_better: True, если меньшее лучше
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    df = results.copy().reset_index(drop=True)
    labels = [", ".join(f"{p}={df.loc[i, p]}" for p in param_cols) for i in range(len(df))]
    best = df[score_col].idxmin() if lower_is_better else df[score_col].idxmax()
    colors = [_BEST if i == best else _BASE for i in range(len(df))]
    figw, fig = new_figure((7.6, max(3.2, 0.45 * len(df) + 1.5)))
    ax = fig.add_subplot(111)
    y = np.arange(len(df))
    ax.barh(y, df[score_col].to_numpy(dtype=float), color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    for i, v in enumerate(df[score_col].to_numpy(dtype=float)):
        ax.text(v, i, " " + f"{v:.4f}", va="center", fontsize=7.5, color=INK)
    ax.set_title(title); ax.set_xlabel(xlabel or score_col); ax.set_ylabel("Hyper-parameter combination")
    _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)


def calibration_plot(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Калибровочные кривые",
    colors: Optional[Sequence[str]] = None,
    save: bool = False,
    fig_id: str = "",
) -> MplFig:
    """
    Калибровочные (надёжностные) кривые нескольких моделей.

    :param curves: словарь «модель → (средний предсказанный риск, наблюдаемая частота)»
    :param title: заголовок
    :param colors: цвета кривых
    :param save: сохранять ли фигуру
    :param fig_id: имя файла при сохранении
    :return: обёртка фигуры :class:`MplFig`
    """
    colors = list(colors) if colors is not None else QUALITATIVE
    figw, fig = new_figure((5.6, 5.4))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], ls="--", color=INK, linewidth=1.2, label="идеальная калибровка")
    for i, (name, (mean_pred, frac_pos)) in enumerate(curves.items()):
        ax.plot(np.asarray(mean_pred), np.asarray(frac_pos), "-o", color=colors[i % len(colors)],
                label=name, markersize=4, markeredgecolor="white", markeredgewidth=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title); ax.set_xlabel("Средний предсказанный риск")
    ax.set_ylabel("Наблюдаемая частота разжижения")
    _legend(ax, fontsize=8); _style_axis(ax); fig.tight_layout()
    return save_figure(figw, fig_id, save)
