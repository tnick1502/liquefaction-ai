"""
Сериализация артефакта синтетической популяции.

Артефакт сохраняется как самодостаточный каталог:
- ``meta.parquet``        — табличные метаданные сценариев;
- ``arrays.npz``          — все числовые массивы (траектории, признаки, индексы сплитов);
- ``config.json``         — конфигурация эксперимента, при которой сгенерированы данные;
- ``feature_names.json``  — имена статических, префиксных и последовательностных признаков.

Такой формат позволяет ноутбуку серии 01 один раз подготовить данные, а ноутбукам
серий 02–04 многократно загружать готовый массив без повторной генерации.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from liquefaction_ai.config import ExperimentConfig

__all__ = ["save_population_artifact", "load_population_artifact"]

_BENCHMARK_KEYS = ("benchmark_idx", "train_rel", "val_rel", "test_rel")
_CRR_KEYS = ("hyperbolic", "power", "exponential", "logarithmic")
_NON_ARRAY_KEYS = (
    "meta",
    "benchmark",
    "crr_families",
    "static_feature_names",
    "prefix_summary_names",
    "seq_feature_names",
    "cohort_filter_counts",
)


def save_population_artifact(artifact_dir: str | Path, population: Dict[str, Any], config: ExperimentConfig) -> None:
    """
    Сохранить артефакт синтетической популяции на диск.

    Числовые массивы записываются в сжатый ``arrays.npz``, табличные метаданные —
    в ``meta.parquet``, вложенные словари (индексы benchmark-сплитов и семейства CRR)
    раскладываются по плоским ключам с префиксами ``benchmark_*`` и ``crr_family_*``.

    :param artifact_dir: каталог для сохранения артефакта (создаётся при отсутствии)
    :param population: словарь популяции, возвращаемый ``generate_population``
    :param config: конфигурация эксперимента, при которой сгенерированы данные
    :return: None
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    population["meta"].to_parquet(artifact_dir / "meta.parquet")
    (artifact_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    arrays: Dict[str, np.ndarray] = {}
    for key, value in population.items():
        if key in _NON_ARRAY_KEYS:
            continue
        if isinstance(value, np.ndarray):
            arrays[key] = value

    bench = population["benchmark"]
    for bk in _BENCHMARK_KEYS:
        arrays[f"benchmark_{bk}"] = bench[bk]

    crr_f = population.get("crr_families")
    if crr_f is not None:
        for ck in _CRR_KEYS:
            arrays[f"crr_family_{ck}"] = crr_f[ck]

    np.savez_compressed(artifact_dir / "arrays.npz", **arrays)

    feature_names = {
        "static_feature_names": population["static_feature_names"],
        "prefix_summary_names": population["prefix_summary_names"],
        "seq_feature_names": population["seq_feature_names"],
    }
    (artifact_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    if "cohort_filter_counts" in population:
        (artifact_dir / "cohort_filter_counts.json").write_text(
            json.dumps(population["cohort_filter_counts"], indent=2), encoding="utf-8"
        )


def load_population_artifact(artifact_dir: str | Path) -> Tuple[Dict[str, Any], ExperimentConfig]:
    """
    Загрузить артефакт синтетической популяции с диска.

    Операция обратна :func:`save_population_artifact`: восстанавливаются табличные
    метаданные, числовые массивы и вложенные словари (индексы benchmark-сплитов и
    семейства CRR), а также конфигурация эксперимента.

    :param artifact_dir: каталог с ранее сохранённым артефактом
    :return: кортеж из словаря популяции и восстановленной конфигурации эксперимента
    """
    artifact_dir = Path(artifact_dir)
    raw = np.load(artifact_dir / "arrays.npz")

    def arr(name: str) -> np.ndarray:
        return raw[name]

    config_dict = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    config = ExperimentConfig(**config_dict)

    feature_names = json.loads((artifact_dir / "feature_names.json").read_text(encoding="utf-8"))

    benchmark = {bk: arr(f"benchmark_{bk}") for bk in _BENCHMARK_KEYS}
    skip = {f"benchmark_{bk}" for bk in _BENCHMARK_KEYS} | {f"crr_family_{ck}" for ck in _CRR_KEYS}

    population: Dict[str, Any] = {
        "meta": pd.read_parquet(artifact_dir / "meta.parquet"),
        "benchmark": benchmark,
        "static_feature_names": feature_names["static_feature_names"],
        "prefix_summary_names": feature_names["prefix_summary_names"],
        "seq_feature_names": feature_names["seq_feature_names"],
    }
    cohort_path = artifact_dir / "cohort_filter_counts.json"
    if cohort_path.exists():
        population["cohort_filter_counts"] = json.loads(cohort_path.read_text(encoding="utf-8"))

    if all(f"crr_family_{ck}" in raw.files for ck in _CRR_KEYS):
        population["crr_families"] = {ck: arr(f"crr_family_{ck}") for ck in _CRR_KEYS}

    for key in raw.files:
        if key in skip:
            continue
        population[key] = raw[key]

    return population, config
