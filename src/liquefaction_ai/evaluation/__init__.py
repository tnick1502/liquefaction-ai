"""
Подпакет оценки качества.

Объединяет полный набор метрик, групповых агрегатов и утилит экспериментов
(``metrics``) и базовые регрессионные метрики общего назначения (``regression``).
"""

from liquefaction_ai.evaluation.metrics import (
    METRIC_COLUMN_TRANSLATIONS,
    METRICS,
    MODEL_DISPLAY_NAMES,
    MetricInfo,
    collect_outputs,
    compute_metrics,
    apply_temperature,
    english_metric_table,
    expected_calibration_error,
    filter_split,
    fit_temperature,
    grouped_metrics,
    is_holdout_region,
    list_metrics,
    localize_metric_table,
    localize_model_names_in_df,
    metric_info,
    metrics_catalog,
    rank_by_metric,
    resolve_nliq_prediction,
    run_quick_experiment,
    safe_binary_metrics,
    subsample_split,
)
from liquefaction_ai.evaluation.regression import compute_mse, compute_r2, compute_wape

__all__ = [
    "collect_outputs",
    "compute_metrics",
    "expected_calibration_error",
    "filter_split",
    "grouped_metrics",
    "is_holdout_region",
    "localize_metric_table",
    "english_metric_table",
    "localize_model_names_in_df",
    "resolve_nliq_prediction",
    "run_quick_experiment",
    "safe_binary_metrics",
    "fit_temperature",
    "apply_temperature",
    "subsample_split",
    "MODEL_DISPLAY_NAMES",
    "METRIC_COLUMN_TRANSLATIONS",
    "MetricInfo",
    "METRICS",
    "metrics_catalog",
    "metric_info",
    "list_metrics",
    "rank_by_metric",
    "compute_r2",
    "compute_mse",
    "compute_wape",
]
