"""
Подпакет визуализации (matplotlib, журнальный стиль Q1).

Объединяет единую пастельную тему (``theme``), обёртку фигур и сохранение в ``results/figs``
в высоком разрешении PNG/PDF (``figure_io``), библиотеку построителей графиков с флагом
``save`` (``plots``) и построители аналитических кривых CRR/PPR (``curves``).
"""

from liquefaction_ai.viz.curves import plot_curves_overlay, plot_function
from liquefaction_ai.viz.panels import (
    crr_physics_panel,
    data_overview_panel,
    model_leaderboard_panel,
)
from liquefaction_ai.viz.figure_io import (
    MplFig,
    find_repo_root,
    new_figure,
    resolve_results_dir,
    save_figure,
)
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
    DIVERGING,
    DIVERGING_NAME,
    GRID,
    INK,
    QUALITATIVE,
    SEQUENTIAL,
    SEQUENTIAL_NAME,
    TEMPLATE_NAME,
    load_color_map,
    model_color_map,
    plain_log_axis,
    register_theme,
    soil_color_map,
)

__all__ = [
    "register_theme",
    "plain_log_axis",
    "TEMPLATE_NAME",
    "QUALITATIVE",
    "SEQUENTIAL",
    "SEQUENTIAL_NAME",
    "DIVERGING",
    "DIVERGING_NAME",
    "INK",
    "GRID",
    "soil_color_map",
    "load_color_map",
    "model_color_map",
    "save_figure",
    "MplFig",
    "new_figure",
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
    "data_overview_panel",
    "crr_physics_panel",
    "model_leaderboard_panel",
]
