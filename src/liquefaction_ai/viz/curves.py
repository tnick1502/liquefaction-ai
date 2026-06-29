"""
Визуализация модельных кривых (matplotlib).

Универсальные построители аналитических функций: одиночная кривая модели с опциональным
наложением экспериментальных точек и наложение нескольких кривых на одной сетке по X с
поддержкой второй оси Y (например, PPR/g(N) слева и поровое давление u в кПа справа) и
логарифмических шкал.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from liquefaction_ai.viz.figure_io import MplFig, new_figure
from liquefaction_ai.viz.theme import GRID, QUALITATIVE, plain_log_axis


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
    scatter_label: str = "Experimental points",
) -> MplFig:
    """
    Несколько модельных кривых на одном графике (общая сетка по оси X).

    Каждый элемент ``curves`` — словарь с ключами ``func`` (функция; первый аргумент с именем
    ``x_arg_name`` — массив X), ``func_params`` (именованные параметры без X), ``label`` и
    опционально ``secondary_y`` (если True — ось Y справа).

    :return: обёртка фигуры :class:`MplFig`
    """
    x_values = _build_x_grid(x_range, n_points, log_x)
    figw, fig = new_figure((7.6, 4.6))
    ax = fig.add_subplot(111)
    any_secondary = any(bool(c.get("secondary_y")) for c in curves)
    ax2 = ax.twinx() if any_secondary else None
    if ax2 is not None:
        ax2.grid(False)

    for i, curve in enumerate(curves):
        func: Callable[..., Union[np.ndarray, float]] = curve["func"]
        params: Dict[str, Any] = dict(curve.get("func_params", {}))
        label = str(curve.get("label", f"model_{i}"))
        target = ax2 if curve.get("secondary_y") else ax
        y_values = np.asarray(func(**{x_arg_name: x_values, **params}), dtype=np.float64)
        target.plot(x_values, y_values, color=QUALITATIVE[i % len(QUALITATIVE)], linewidth=2.0, label=label)

    if scatter_points is not None:
        xp, yp = scatter_points
        ax.scatter(np.asarray(xp), np.asarray(yp), s=34, color="#2b2f36", zorder=5,
                   edgecolors="white", linewidths=0.4, label=scatter_label)

    ax.set_title(title or ""); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    if log_x: ax.set_xscale("log"); plain_log_axis(ax, "x")
    if log_y: ax.set_yscale("log"); plain_log_axis(ax, "y")
    if ax2 is not None:
        if y_label_secondary: ax2.set_ylabel(y_label_secondary)
        if log_y_secondary: ax2.set_yscale("log"); plain_log_axis(ax2, "y")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if ax2 is not None else ([], []))
    leg = ax.legend(h1 + h2, l1 + l2, loc="best", frameon=True, framealpha=0.85, edgecolor=GRID, fontsize=8.5)
    if leg: leg.get_frame().set_linewidth(0.7)
    for side in ("top",):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return figw


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
    func_label: str = "Model",
    scatter_label: str = "Experimental points",
    log_x: bool = False,
    log_y: bool = False,
) -> MplFig:
    """
    Построение графика математической функции (matplotlib).

    :param func: функция модели; первый аргумент с именем ``x_arg_name`` — массив X
    :param func_params: именованные параметры (без аргумента X)
    :param x_range: (x_min, x_max)
    :param n_points: число точек кривой
    :param x_arg_name: имя X-аргумента функции
    :param scatter_points: (x_array, y_array) экспериментальных точек
    :param title: заголовок
    :param x_label: подпись оси X
    :param y_label: подпись оси Y
    :param func_label: подпись кривой
    :param scatter_label: подпись точек
    :param log_x: лог-шкала X
    :param log_y: лог-шкала Y
    :return: обёртка фигуры :class:`MplFig`
    """
    x_values = _build_x_grid(x_range, n_points, log_x)
    y_values = np.asarray(func(**{x_arg_name: x_values, **func_params}), dtype=np.float64)
    figw, fig = new_figure((7.4, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(x_values, y_values, color=QUALITATIVE[0], linewidth=2.2, label=func_label)
    if scatter_points is not None:
        xp, yp = scatter_points
        ax.scatter(np.asarray(xp), np.asarray(yp), s=34, color="#2b2f36", zorder=5,
                   edgecolors="white", linewidths=0.4, label=scatter_label)
    ax.set_title(title or ""); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    if log_x: ax.set_xscale("log"); plain_log_axis(ax, "x")
    if log_y: ax.set_yscale("log"); plain_log_axis(ax, "y")
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    leg = ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor=GRID, fontsize=8.5)
    if leg: leg.get_frame().set_linewidth(0.7)
    fig.tight_layout()
    return figw
