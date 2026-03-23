from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go


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
