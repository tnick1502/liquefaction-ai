"""
Библиотека построителей графиков Plotly для ноутбуков проекта.

Каждый построитель создаёт оформленную в едином стиле интерактивную фигуру и принимает
флаг ``save``: при ``save=True`` фигура сохраняется в ``results/figs/`` под именем ``fig_id``
(см. :func:`liquefaction_ai.viz.figure_io.save_figure`). Набор покрывает распределения,
box-плоты по группам, корреляционные карты, диаграммы рассеяния, трёхмерные поверхности,
столбчатые диаграммы, линейные графики (кривые обучения и траектории), полосы
неопределённости и калибровочные кривые.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

from liquefaction_ai.viz.figure_io import save_figure
from liquefaction_ai.viz.theme import QUALITATIVE, SEQUENTIAL

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
) -> go.Figure:
    """
    Мультипанельный дашборд результатов grid search: по оси Y — метрики, по оси X —
    текстовое описание конфигурации гиперпараметров.

    Каждая метрика выводится отдельной панелью (общая ось X с подписями конфигураций);
    лучшая по ``selected_metric`` конфигурация подсвечивается. Это даёт «живую» и
    информативную историю перебора сразу по многим параметрам и метрикам.

    :param results: таблица результатов grid search (DataFrame со столбцами параметров и метрик)
    :param metric_keys: список ключей метрик для отображения (по одной панели на метрику)
    :param param_cols: столбцы гиперпараметров для текстовой подписи конфигураций
    :param selected_metric: метрика, по которой выбрана лучшая конфигурация (для подсветки)
    :param metric_labels: подписи панелей «ключ → название (единицы)» (по умолчанию ключ)
    :param metric_fmts: форматы чисел по метрикам (по умолчанию ``.4f``)
    :param lower_is_better: направление ``selected_metric`` (меньше — лучше)
    :param target: целевое значение ``selected_metric`` (если задано, лучший — по близости к нему)
    :param title: общий заголовок фигуры
    :param row_height: высота одной панели, пикс.
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    df = results.reset_index(drop=True)
    metric_labels = metric_labels or {}
    metric_fmts = metric_fmts or {}
    labels = [", ".join(f"{p}={df.loc[i, p]}" for p in param_cols) for i in range(len(df))]

    if target is not None:
        best_idx = int((df[selected_metric] - target).abs().idxmin())
    else:
        best_idx = int(df[selected_metric].idxmin() if lower_is_better else df[selected_metric].idxmax())

    n = len(metric_keys)
    subtitles = [metric_labels.get(k, k) for k in metric_keys]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=subtitles, vertical_spacing=0.06)
    for j, key in enumerate(metric_keys):
        colors = ["#198754" if i == best_idx else "#0b6efd" for i in range(len(df))]
        fmt = metric_fmts.get(key, ".4f")
        text = [format(v, fmt) if np.isfinite(v) else "" for v in df[key].to_numpy(dtype=float)]
        fig.add_trace(
            go.Bar(x=labels, y=df[key], marker_color=colors, text=text, textposition="auto", showlegend=False),
            row=j + 1, col=1,
        )
        fig.update_yaxes(title_text="value", row=j + 1, col=1)
    fig.update_xaxes(title_text="Hyper-parameter configuration", row=n, col=1, tickangle=-25)
    fig.update_layout(title=f"{title} (best by {selected_metric} highlighted)", height=row_height * n)
    return save_figure(fig, fig_id, save)


def heatmap(
    matrix,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str = "",
    colorscale: str = "Viridis",
    value_fmt: Optional[str] = ".2f",
    colorbar_title: str = "",
    height: int = 520,
    save: bool = False,
    fig_id: str = "",
) -> go.Figure:
    """
    Аннотированная тепловая карта произвольной матрицы (не обязательно квадратной).

    Удобна для отображения средних значений по группам (например, вклад факторов CRR
    по типам грунта).

    :param matrix: двумерный массив значений, форма (len(y_labels), len(x_labels))
    :param x_labels: подписи столбцов
    :param y_labels: подписи строк
    :param title: заголовок фигуры
    :param colorscale: цветовая шкала Plotly
    :param value_fmt: формат числовых аннотаций (или None — без аннотаций)
    :param colorbar_title: подпись цветовой шкалы
    :param height: высота фигуры, пикс.
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    z = np.asarray(matrix, dtype=float)
    x_labels = list(x_labels)
    y_labels = list(y_labels)
    fig = go.Figure(go.Heatmap(z=z, x=x_labels, y=y_labels, colorscale=colorscale,
                               colorbar=dict(title=colorbar_title)))
    if value_fmt:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                fig.add_annotation(x=x_labels[j], y=y_labels[i], text=format(z[i, j], value_fmt),
                                   showarrow=False, font=dict(size=8, color="#1f2937"))
    fig.update_layout(title=title, height=height, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(tickangle=35)
    return save_figure(fig, fig_id, save)


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
) -> go.Figure:
    """
    Сетка гистограмм распределений выбранных колонок таблицы.

    :param df: таблица данных
    :param columns: список колонок для построения гистограмм
    :param titles: заголовки подграфиков (по умолчанию — имена колонок)
    :param n_cols: число столбцов в сетке подграфиков
    :param colors: цвета гистограмм (по умолчанию — качественная палитра)
    :param nbins: число интервалов гистограммы
    :param title: общий заголовок фигуры
    :param row_height: высота одного ряда подграфиков, пикс.
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    titles = list(titles) if titles is not None else list(columns)
    colors = list(colors) if colors is not None else QUALITATIVE
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.08)
    for i, col in enumerate(columns):
        r, c = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(
            go.Histogram(x=np.asarray(df[col]), nbinsx=nbins, marker_color=colors[i % len(colors)],
                         marker_line_color="white", marker_line_width=0.3, opacity=0.88, showlegend=False),
            row=r, col=c,
        )
    fig.update_layout(title=title, height=row_height * n_rows, bargap=0.05)
    return save_figure(fig, fig_id, save)


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
) -> go.Figure:
    """
    Сетка box-плотов: распределение каждой величины в разрезе групп.

    :param df: таблица данных
    :param value_cols: список числовых колонок (по одной на подграфик)
    :param group_col: колонка группировки (например, тип грунта)
    :param group_order: порядок групп на оси X
    :param group_labels: подписи групп (по умолчанию — исходные значения)
    :param titles: заголовки подграфиков (по умолчанию — имена колонок)
    :param n_cols: число столбцов в сетке подграфиков
    :param colors: цвета групп (по умолчанию — качественная палитра)
    :param title: общий заголовок фигуры
    :param row_height: высота одного ряда подграфиков, пикс.
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    titles = list(titles) if titles is not None else list(value_cols)
    colors = list(colors) if colors is not None else QUALITATIVE
    labels = {g: (group_labels or {}).get(g, g) for g in group_order}
    n_rows = int(np.ceil(len(value_cols) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles, vertical_spacing=0.13, horizontal_spacing=0.07)
    for i, col in enumerate(value_cols):
        r, c = i // n_cols + 1, i % n_cols + 1
        for gi, g in enumerate(group_order):
            vals = df.loc[df[group_col] == g, col].to_numpy()
            fig.add_trace(
                go.Box(y=vals, name=labels[g], legendgroup=labels[g], showlegend=(i == 0),
                       marker_color=colors[gi % len(colors)], boxmean=True, line_width=1.4),
                row=r, col=c,
            )
    fig.update_layout(title=title, height=row_height * n_rows, boxmode="group")
    fig.update_xaxes(showticklabels=False)
    return save_figure(fig, fig_id, save)


def correlation_heatmap(
    corr,
    title: str = "Корреляционная матрица",
    save: bool = False,
    fig_id: str = "",
    height: int = 720,
) -> go.Figure:
    """
    Тепловая карта корреляционной матрицы с числовыми аннотациями.

    :param corr: квадратная таблица корреляций (pandas DataFrame)
    :param title: заголовок фигуры
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :param height: высота фигуры, пикс.
    :return: фигура Plotly
    """
    labels = list(corr.columns)
    z = np.asarray(corr.to_numpy(), dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
            colorbar=dict(title="r"), hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
        )
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(x=labels[j], y=labels[i], text=f"{z[i, j]:.2f}", showarrow=False,
                               font=dict(size=8, color="white" if abs(z[i, j]) > 0.5 else "#1f2937"))
    fig.update_layout(title=title, height=height, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(tickangle=50)
    return save_figure(fig, fig_id, save)


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
) -> go.Figure:
    """
    Диаграмма рассеяния с опциональной непрерывной окраской точек.

    :param x: значения по оси X
    :param y: значения по оси Y
    :param color: значения для непрерывной окраски (или None)
    :param color_label: подпись цветовой шкалы
    :param title: заголовок фигуры
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param logx: логарифмическая шкала по X
    :param logy: логарифмическая шкала по Y
    :param hline: уровень горизонтальной опорной линии (или None)
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    marker = dict(size=5, opacity=0.6)
    if color is not None:
        marker.update(color=np.asarray(color), colorscale=SEQUENTIAL, showscale=True,
                      colorbar=dict(title=color_label))
    else:
        marker.update(color=QUALITATIVE[0])
    fig = go.Figure(go.Scattergl(x=np.asarray(x), y=np.asarray(y), mode="markers", marker=marker))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#dc3545")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    if logx:
        fig.update_xaxes(type="log")
    if logy:
        fig.update_yaxes(type="log")
    return save_figure(fig, fig_id, save)


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
) -> go.Figure:
    """
    Диаграмма рассеяния, раскрашенная по категориальной группе.

    :param df: таблица данных
    :param x: имя колонки оси X
    :param y: имя колонки оси Y
    :param group_col: колонка категориальной группы
    :param group_order: порядок групп
    :param group_labels: подписи групп
    :param colors: цвета групп (по умолчанию — качественная палитра)
    :param title: заголовок фигуры
    :param xlabel: подпись оси X (по умолчанию имя колонки)
    :param ylabel: подпись оси Y (по умолчанию имя колонки)
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    colors = list(colors) if colors is not None else QUALITATIVE
    labels = {g: (group_labels or {}).get(g, g) for g in group_order}
    fig = go.Figure()
    for gi, g in enumerate(group_order):
        sub = df[df[group_col] == g]
        fig.add_trace(go.Scattergl(x=sub[x], y=sub[y], mode="markers", name=labels[g],
                                   marker=dict(size=5, opacity=0.55, color=colors[gi % len(colors)])))
    fig.update_layout(title=title, xaxis_title=xlabel or x, yaxis_title=ylabel or y)
    return save_figure(fig, fig_id, save)


def _interp_grid(x, y, z, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Интерполировать рассеянные точки на регулярную сетку (линейно + ближайший сосед).

    :param x: значения первой оси
    :param y: значения второй оси
    :param z: значения целевой величины
    :param resolution: число узлов сетки по каждой оси
    :return: кортеж (xx, yy, zz)
    """
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
) -> go.Figure:
    """
    Сетка интерполированных трёхмерных поверхностей.

    :param specs: список словарей-спецификаций, каждый содержит ключи
                  ``x``/``y``/``z`` (данные), ``title``, ``xlabel``/``ylabel``/``zlabel`` и
                  опционально ``colorscale``
    :param n_cols: число поверхностей в ряду
    :param resolution: разрешение интерполяционной сетки
    :param title: общий заголовок фигуры
    :param height: высота одного ряда, пикс.
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    n = len(specs)
    n_rows = int(np.ceil(n / n_cols))
    fig = make_subplots(
        rows=n_rows, cols=n_cols, specs=[[{"type": "surface"}] * n_cols for _ in range(n_rows)],
        subplot_titles=[s.get("title", "") for s in specs], horizontal_spacing=0.04, vertical_spacing=0.08,
    )
    for i, s in enumerate(specs):
        r, c = i // n_cols + 1, i % n_cols + 1
        xx, yy, zz = _interp_grid(s["x"], s["y"], s["z"], resolution)
        fig.add_trace(
            go.Surface(x=xx, y=yy, z=zz, colorscale=s.get("colorscale", SEQUENTIAL), showscale=False,
                       contours_z=dict(show=True, usecolormap=True, project_z=True)),
            row=r, col=c,
        )
        scene_id = "scene" if i == 0 else f"scene{i + 1}"
        fig.layout[scene_id].update(
            xaxis_title=s.get("xlabel", "x"), yaxis_title=s.get("ylabel", "y"), zaxis_title=s.get("zlabel", "z"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        )
    fig.update_layout(title=title, height=height * n_rows, margin=dict(l=10, r=10, t=70, b=10))
    return save_figure(fig, fig_id, save)


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
) -> go.Figure:
    """
    Столбчатая диаграмма (вертикальная или горизонтальная).

    :param categories: подписи категорий
    :param values: значения столбцов
    :param title: заголовок фигуры
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param color: цвет столбцов
    :param horizontal: горизонтальная ориентация
    :param text_fmt: формат подписи значений (или None — без подписей)
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    values = np.asarray(values, dtype=float)
    text = [format(v, text_fmt) if text_fmt and np.isfinite(v) else "" for v in values]
    if horizontal:
        fig = go.Figure(go.Bar(y=list(categories), x=values, orientation="h", marker_color=color,
                               text=text, textposition="auto"))
    else:
        fig = go.Figure(go.Bar(x=list(categories), y=values, marker_color=color, text=text, textposition="auto"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    return save_figure(fig, fig_id, save)


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
) -> go.Figure:
    """
    Сгруппированная столбчатая диаграмма для сравнения нескольких рядов.

    :param categories: подписи категорий по оси X
    :param series: словарь «имя ряда → значения»
    :param title: заголовок фигуры
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param colors: цвета рядов (по умолчанию — качественная палитра)
    :param barmode: режим столбцов (``group`` или ``stack``)
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    colors = list(colors) if colors is not None else QUALITATIVE
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(name=name, x=list(categories), y=np.asarray(vals, dtype=float),
                             marker_color=colors[i % len(colors)]))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, barmode=barmode)
    return save_figure(fig, fig_id, save)


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
) -> go.Figure:
    """
    Линейный график из нескольких рядов (кривые обучения, траектории и т.п.).

    :param series: список рядов; каждый ряд — словарь с ключами ``x``, ``y``, ``name`` и
                   опционально ``mode`` (``lines``/``lines+markers``), ``color``, ``dash``, ``width``
    :param title: заголовок фигуры
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param logx: логарифмическая шкала по X
    :param logy: логарифмическая шкала по Y
    :param hline: уровень горизонтальной опорной линии (или None)
    :param height: высота фигуры, пикс.
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    fig = go.Figure()
    for i, s in enumerate(series):
        fig.add_trace(
            go.Scatter(
                x=np.asarray(s["x"]), y=np.asarray(s["y"]), name=s.get("name", f"ряд {i}"),
                mode=s.get("mode", "lines"),
                line=dict(color=s.get("color", QUALITATIVE[i % len(QUALITATIVE)]),
                          dash=s.get("dash", "solid"), width=s.get("width", 2.4)),
            )
        )
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#dc3545")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, height=height)
    if logx:
        fig.update_xaxes(type="log")
    if logy:
        fig.update_yaxes(type="log")
    return save_figure(fig, fig_id, save)


def line_with_bands(
    x,
    mean,
    lower,
    upper,
    extra: Optional[Sequence[Dict]] = None,
    band_color: str = "rgba(11,110,253,0.18)",
    line_color: str = "#0b6efd",
    mean_name: str = "Среднее",
    band_name: str = "90% интервал",
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    logx: bool = False,
    save: bool = False,
    fig_id: str = "",
) -> go.Figure:
    """
    Линия со средним и закрашенной полосой неопределённости.

    :param x: значения по оси X
    :param mean: среднее предсказание
    :param lower: нижняя граница интервала
    :param upper: верхняя граница интервала
    :param extra: дополнительные ряды (список словарей ``x``/``y``/``name``/``color``/``dash``)
    :param band_color: цвет заливки полосы (rgba)
    :param line_color: цвет линии среднего
    :param mean_name: подпись линии среднего
    :param band_name: подпись полосы интервала
    :param title: заголовок фигуры
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param logx: логарифмическая шкала по X
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    x = np.asarray(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([np.asarray(upper), np.asarray(lower)[::-1]]),
                             fill="toself", fillcolor=band_color, line=dict(color="rgba(0,0,0,0)"),
                             name=band_name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=np.asarray(mean), mode="lines", name=mean_name,
                             line=dict(color=line_color, width=2.6)))
    for s in (extra or []):
        fig.add_trace(go.Scatter(x=np.asarray(s["x"]), y=np.asarray(s["y"]), mode=s.get("mode", "lines"),
                                 name=s.get("name", ""), line=dict(color=s.get("color", "#1f2937"),
                                 dash=s.get("dash", "solid"), width=s.get("width", 2.2))))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    if logx:
        fig.update_xaxes(type="log")
    return save_figure(fig, fig_id, save)


def training_dashboard(
    history,
    title: str = "Training dynamics",
    model_color: str = "#0b6efd",
    save: bool = False,
    fig_id: str = "",
    height: int = 420,
) -> go.Figure:
    """
    Дашборд динамики обучения: функция потерь и валидационные метрики по эпохам.

    Строит мультипанельную фигуру: потери (train/val), а также доступные валидационные
    метрики (AUROC, RMSE траектории), что делает кривые обучения «живыми» и информативными.

    :param history: история обучения (DataFrame с колонкой ``epoch``, ``train_loss``,
                    ``val_loss`` и опционально ``val_auroc``/``val_traj_rmse``/``val_brier``)
    :param title: общий заголовок фигуры
    :param model_color: базовый цвет модели
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :param height: высота фигуры, пикс.
    :return: фигура Plotly
    """
    epoch = np.asarray(history["epoch"])
    panels = [("loss", "Loss (–)")]
    if "val_auroc" in history:
        panels.append(("val_auroc", "Validation AUROC (–)"))
    if "val_traj_rmse" in history:
        panels.append(("val_traj_rmse", "Validation trajectory RMSE (–)"))
    n = len(panels)
    titles = {"loss": "Loss", "val_auroc": "Validation AUROC",
              "val_traj_rmse": "Validation trajectory RMSE"}
    fig = make_subplots(rows=1, cols=n, subplot_titles=[titles[k] for k, _ in panels],
                        horizontal_spacing=0.08)
    for j, (key, ylab) in enumerate(panels):
        col = j + 1
        if key == "loss":
            fig.add_trace(go.Scatter(x=epoch, y=history["train_loss"], name="train loss",
                                     mode="lines+markers", line=dict(color=model_color, width=2.6)), row=1, col=col)
            fig.add_trace(go.Scatter(x=epoch, y=history["val_loss"], name="val loss",
                                     mode="lines+markers", line=dict(color="#d63384", width=2.6, dash="dash")), row=1, col=col)
        else:
            fig.add_trace(go.Scatter(x=epoch, y=history[key], name=titles[key], showlegend=False,
                                     mode="lines+markers", line=dict(color="#198754", width=2.6)), row=1, col=col)
        fig.update_xaxes(title_text="Epoch", row=1, col=col)
        fig.update_yaxes(title_text=ylab, row=1, col=col)
    fig.update_layout(title=title, height=height)
    return save_figure(fig, fig_id, save)


def grid_search_plot(
    results,
    score_col: str,
    param_cols: Sequence[str],
    title: str = "Grid search results",
    xlabel: Optional[str] = None,
    lower_is_better: bool = True,
    save: bool = False,
    fig_id: str = "",
) -> go.Figure:
    """
    Горизонтальная диаграмма результатов перебора гиперпараметров.

    Каждой комбинации соответствует столбец со значением метрики отбора; лучшая
    комбинация выделяется цветом.

    :param results: таблица результатов grid search (DataFrame)
    :param score_col: имя колонки с метрикой отбора
    :param param_cols: колонки гиперпараметров для подписи комбинаций
    :param title: заголовок фигуры
    :param xlabel: подпись оси значений (по умолчанию — имя метрики)
    :param lower_is_better: True, если меньшее значение метрики лучше (определяет лучший столбец)
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    df = results.copy().reset_index(drop=True)
    labels = [", ".join(f"{p}={df.loc[i, p]}" for p in param_cols) for i in range(len(df))]
    best_idx = df[score_col].idxmin() if lower_is_better else df[score_col].idxmax()
    colors = ["#198754" if i == best_idx else "#0b6efd" for i in range(len(df))]
    fig = go.Figure(go.Bar(x=df[score_col], y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:.4f}" for v in df[score_col]], textposition="auto"))
    fig.update_layout(title=title, xaxis_title=xlabel or score_col, yaxis_title="Hyper-parameter combination",
                      yaxis=dict(autorange="reversed"))
    return save_figure(fig, fig_id, save)


def calibration_plot(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Калибровочные кривые",
    colors: Optional[Sequence[str]] = None,
    save: bool = False,
    fig_id: str = "",
) -> go.Figure:
    """
    Калибровочные (надёжностные) кривые нескольких моделей.

    :param curves: словарь «имя модели → (средний предсказанный риск, наблюдаемая частота)»
    :param title: заголовок фигуры
    :param colors: цвета кривых (по умолчанию — качественная палитра)
    :param save: сохранять ли фигуру в ``results/figs``
    :param fig_id: имя файла при сохранении
    :return: фигура Plotly
    """
    colors = list(colors) if colors is not None else QUALITATIVE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="идеальная калибровка",
                             line=dict(color="#1f2937", dash="dash", width=1.4)))
    for i, (name, (mean_pred, frac_pos)) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=np.asarray(mean_pred), y=np.asarray(frac_pos), mode="lines+markers",
                                 name=name, line=dict(color=colors[i % len(colors)], width=2.4)))
    fig.update_layout(title=title, xaxis_title="Средний предсказанный риск",
                      yaxis_title="Наблюдаемая частота разжижения",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return save_figure(fig, fig_id, save)
