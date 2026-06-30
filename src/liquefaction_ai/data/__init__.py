"""
Подпакет работы с данными.

Объединяет генерацию синтетической популяции (``synthetic``), стратифицированное
разбиение и сборку benchmark-тензоров (``splits``), сериализацию артефакта (``io``), локализацию метаданных (``meta``)
и сборку популяции из сырых пиклов опытов (``raw_loader``).
"""

from liquefaction_ai.data.io import load_population_artifact, save_population_artifact
from liquefaction_ai.data.meta import localize_meta_frame, localize_series
from liquefaction_ai.data.ppr_envelope import (
    extract_upper_envelope,
    monotone_smooth,
    smooth_ppr_trajectory,
)
from liquefaction_ai.data.raw_loader import (build_real_objects_population, find_cloud_root,
                                              DEFAULT_TEST_TYPES, read_statement, build_cohort_manifest)
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
    "read_statement",
    "DEFAULT_TEST_TYPES",
    "find_cloud_root",
    "build_real_objects_population",
    "build_cohort_manifest",
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
    "smooth_ppr_trajectory",
    "extract_upper_envelope",
    "monotone_smooth",
]
