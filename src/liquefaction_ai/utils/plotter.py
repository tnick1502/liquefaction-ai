from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go

_DEFAULT_COLORS: Tuple[str, ...] = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def _build_x_grid(x_range: Tuple[float, float], n_points: int, log_x: bool) -> np.ndarray:
    x_min, x_max = x_range
    if log_x:
        return np.logspace(np.log10(max(x_min, 1e-12)), np.log10(x_max), n_points)
    return np.linspace(x_min, x_max, n_points)


def plot_curves_overlay(
    curves: Sequence[Dict[str, Any]],
    x_range: Tuple[float, float],
    n_points: int = 500,
    x_arg_name: str = "n_cycles",
    title: Optional[str] = None,
    x_label: str = "X",
    y_label: str = "Y",
    y_label_secondary: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
    log_y_secondary: bool = False,
    scatter_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    scatter_label: str = "Экспериментальные точки",
) -> go.Figure:
    """
    Несколько модельных кривых на одном графике (общая сетка по оси X).

    Каждый элемент ``curves`` — словарь с ключами:
    - ``func`` — вызываемая функция (первый аргумент с именем ``x_arg_name`` — массив X);
    - ``func_params`` — именованные параметры без X;
    - ``label`` — подпись в легенде;
    - ``secondary_y`` (опционально) — если True, ось Y справа (например u в кПа рядом с PPR).
    """
    x_values = _build_x_grid(x_range, n_points, log_x)
    fig = go.Figure()
    any_secondary = any(bool(c.get("secondary_y")) for c in curves)

    for i, curve in enumerate(curves):
        func: Callable[..., Union[np.ndarray, float]] = curve["func"]
        func_params: Dict[str, Any] = dict(curve.get("func_params", {}))
        label = str(curve.get("label", f"model_{i}"))
        secondary = bool(curve.get("secondary_y", False))
        call_kw: Dict[str, Any] = {x_arg_name: x_values, **func_params}
        y_values = np.asarray(func(**call_kw), dtype=np.float64)
        yaxis = "y2" if secondary else "y"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=label,
                line=dict(width=2, color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]),
                yaxis=yaxis,
            )
        )

    if scatter_points is not None:
        x_pts, y_pts = scatter_points
        fig.add_trace(
            go.Scatter(
                x=np.asarray(x_pts),
                y=np.asarray(y_pts),
                mode="markers",
                name=scatter_label,
                marker=dict(size=8, symbol="circle"),
                yaxis="y",
            )
        )

    layout: Dict[str, Any] = {
        "title": title,
        "xaxis_title": x_label,
        "template": "plotly_white",
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    }
    if log_x:
        layout["xaxis_type"] = "log"

    layout["yaxis"] = dict(title=y_label, type="log" if log_y else "linear")

    if any_secondary:
        y2: Dict[str, Any] = dict(overlaying="y", side="right", type="log" if log_y_secondary else "linear")
        if y_label_secondary:
            y2["title"] = y_label_secondary
        layout["yaxis2"] = y2

    fig.update_layout(**layout)
    return fig


def plot_function(
    func: Callable[..., Union[np.ndarray, float]],
    func_params: Dict[str, Any],
    x_range: Tuple[float, float],
    n_points: int = 500,
    x_arg_name: str = "n_cycles",
    scatter_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: Optional[str] = None,
    x_label: str = "X",
    y_label: str = "Y",
    func_label: str = "Модель",
    scatter_label: str = "Экспериментальные точки",
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """
    Построение графика математической функции с помощью Plotly.

    Вычисляет значения переданной функции на равномерной (или логарифмической) сетке
    по оси X, строит линию и, при необходимости, накладывает scatter-точки.

    :param func: вызываемая функция модели; первый позиционный аргумент — массив X
    :param func_params: словарь именованных параметров, передаваемых в func (без аргумента X)
    :param x_range: кортеж (x_min, x_max), задающий диапазон построения по оси X
    :param n_points: количество точек для построения кривой (по умолчанию 500)
    :param x_arg_name: имя аргумента функции, отвечающего за ось X (по умолчанию "n_cycles")
    :param scatter_points: кортеж (x_array, y_array) экспериментальных точек для наложения поверх кривой
    :param title: заголовок графика
    :param x_label: подпись оси X
    :param y_label: подпись оси Y
    :param func_label: подпись кривой модели в легенде
    :param scatter_label: подпись scatter-точек в легенде
    :param log_x: использовать логарифмическую шкалу по оси X
    :param log_y: использовать логарифмическую шкалу по оси Y
    :return: объект plotly.graph_objects.Figure с построенным графиком
    """
    x_min, x_max = x_range

    if log_x:
        x_values: np.ndarray = np.logspace(
            np.log10(max(x_min, 1e-12)),
            np.log10(x_max),
            n_points,
        )
    else:
        x_values = np.linspace(x_min, x_max, n_points)

    call_kwargs: Dict[str, Any] = {x_arg_name: x_values, **func_params}
    y_values: np.ndarray = np.asarray(func(**call_kwargs), dtype=np.float64)

    fig: go.Figure = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=func_label,
            line=dict(width=2),
        )
    )

    if scatter_points is not None:
        x_pts, y_pts = scatter_points
        fig.add_trace(
            go.Scatter(
                x=np.asarray(x_pts),
                y=np.asarray(y_pts),
                mode="markers",
                name=scatter_label,
                marker=dict(size=8, symbol="circle"),
            )
        )

    axis_opts: Dict[str, Any] = {}
    if log_x:
        axis_opts["xaxis_type"] = "log"
    if log_y:
        axis_opts["yaxis_type"] = "log"

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **axis_opts,
    )

    return fig
