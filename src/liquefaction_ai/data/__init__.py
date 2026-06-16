"""
Подпакет работы с данными.

Объединяет генерацию синтетической популяции (``synthetic``), стратифицированное
разбиение и сборку benchmark-тензоров (``splits``), сериализацию артефакта (``io``)
и локализацию метаданных (``meta``).
"""

from liquefaction_ai.data.io import load_population_artifact, save_population_artifact
from liquefaction_ai.data.meta import localize_meta_frame, localize_series
from liquefaction_ai.data.real_adapter import (
    build_observed_prefix,
    build_population_from_experiments,
    compute_crr_features,
)
from liquefaction_ai.data.splits import (
    iterate_minibatches,
    make_benchmark_splits,
    prepare_benchmark_dataset,
    safe_strata,
    stratified_subset_indices,
)
from liquefaction_ai.data.synthetic import generate_population

__all__ = [
    "generate_population",
    "save_population_artifact",
    "load_population_artifact",
    "prepare_benchmark_dataset",
    "make_benchmark_splits",
    "stratified_subset_indices",
    "safe_strata",
    "iterate_minibatches",
    "localize_meta_frame",
    "localize_series",
    "build_population_from_experiments",
    "build_observed_prefix",
    "compute_crr_features",
]
