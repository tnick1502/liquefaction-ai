"""Утилиты: синтетические данные, сплиты, I/O, обучение (лоссы/цикл), оценка, визуализация Plotly, numpy-метрики."""

from liquefaction_ai.utils.evaluation import (
    METRIC_COLUMN_TRANSLATIONS,
    MODEL_DISPLAY_NAMES,
    collect_outputs,
    compute_metrics,
    expected_calibration_error,
    filter_split,
    grouped_metrics,
    is_holdout_region,
    localize_metric_table,
    localize_model_names_in_df,
    resolve_nliq_prediction,
    run_quick_experiment,
    safe_binary_metrics,
    subsample_split,
)
from liquefaction_ai.utils.evaluating import (
    compute_mse,
    compute_r2,
    compute_wape,
)
from liquefaction_ai.utils.io import load_population_artifact, save_population_artifact
from liquefaction_ai.utils.losses import (
    clone_state_dict,
    gaussian_nll,
    masked_mae,
    masked_mean,
    masked_mse,
)
from liquefaction_ai.utils.plotter import plot_curves_overlay, plot_function
from liquefaction_ai.utils.splits import iterate_minibatches, prepare_benchmark_dataset
from liquefaction_ai.utils.synthetic import generate_population
from liquefaction_ai.utils.train_loop import train_model

__all__ = [
    "METRIC_COLUMN_TRANSLATIONS",
    "MODEL_DISPLAY_NAMES",
    "collect_outputs",
    "compute_metrics",
    "compute_mse",
    "compute_r2",
    "compute_wape",
    "expected_calibration_error",
    "filter_split",
    "gaussian_nll",
    "generate_population",
    "grouped_metrics",
    "is_holdout_region",
    "iterate_minibatches",
    "load_population_artifact",
    "localize_metric_table",
    "localize_model_names_in_df",
    "prepare_benchmark_dataset",
    "resolve_nliq_prediction",
    "plot_curves_overlay",
    "plot_function",
    "run_quick_experiment",
    "safe_binary_metrics",
    "save_population_artifact",
    "subsample_split",
    "train_model",
    "clone_state_dict",
    "masked_mean",
    "masked_mae",
    "masked_mse",
]
