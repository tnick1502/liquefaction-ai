"""
Подпакет визуализации (Plotly).

Объединяет единую тему оформления (``theme``), сохранение фигур в ``results/figs``
(``figure_io``), библиотеку построителей графиков с флагом ``save`` (``plots``) и
интерактивные построители аналитических кривых CRR/PPR (``curves``).
"""

from liquefaction_ai.viz.curves import plot_curves_overlay, plot_function
from liquefaction_ai.viz.figure_io import find_repo_root, resolve_results_dir, save_figure
from liquefaction_ai.viz.plots import (
    bar,
    box_grid,
    calibration_plot,
    correlation_heatmap,
    grid_search_dashboard,
    grid_search_plot,
    grouped_bar,
    heatmap,
    histogram_grid,
    line_with_bands,
    lines,
    scatter,
    scatter_by_group,
    surface3d_grid,
    training_dashboard,
)
from liquefaction_ai.viz.theme import (
    QUALITATIVE,
    SEQUENTIAL,
    TEMPLATE_NAME,
    load_color_map,
    model_color_map,
    register_theme,
    soil_color_map,
)

__all__ = [
    "register_theme",
    "TEMPLATE_NAME",
    "QUALITATIVE",
    "SEQUENTIAL",
    "soil_color_map",
    "load_color_map",
    "model_color_map",
    "save_figure",
    "find_repo_root",
    "resolve_results_dir",
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
    "plot_curves_overlay",
    "plot_function",
]
