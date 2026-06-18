"""
Единый выбор источника данных для всего пайплайна.

Проект умеет работать с двумя взаимозаменяемыми датасетами одного формата (артефакт
популяции из :func:`liquefaction_ai.data.io.save_population_artifact`):

* ``synthetic`` — синтетическая популяция (гладкие кривые PPR(N) по построению);
* ``real_objects`` — реальные объекты, собранные ноутбуком ``1_1_3`` (пиклы + ведомость,
  гладкая линия PPR по верхней огибающей).

«Одно место выбора»: достаточно задать имя источника и вызвать :func:`materialize_dataset`
— выбранный артефакт копируется в канонический каталог ``data/demo_run``, который читают все
последующие ноутбуки (анализ → обучение → оценка). Менять сами ноутбуки не нужно.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.data.io import load_population_artifact, save_population_artifact

__all__ = [
    "DATASET_SOURCES",
    "CANONICAL_DIR",
    "dataset_dir",
    "available_sources",
    "materialize_dataset",
    "load_active_population",
]

# Имя источника → относительный (от корня репозитория) путь к его артефакту.
DATASET_SOURCES: Dict[str, str] = {
    "synthetic": "data/demo_source_synthetic",
    "real_objects": "data/real_objects",
}

# Канонический каталог, который читают все ноутбуки анализа/обучения/оценки.
CANONICAL_DIR = "data/demo_run"

_ARTIFACT_MARKERS = ("arrays.npz", "meta.parquet", "config.json")


def dataset_dir(source: str, repo_root: Path) -> Path:
    """
    Путь к каталогу артефакта для именованного источника.

    :param source: имя источника (ключ :data:`DATASET_SOURCES`)
    :param repo_root: корень репозитория
    :return: абсолютный путь к каталогу артефакта источника
    """
    if source not in DATASET_SOURCES:
        raise KeyError(f"Неизвестный источник '{source}'. Доступны: {sorted(DATASET_SOURCES)}")
    return Path(repo_root) / DATASET_SOURCES[source]


def _is_artifact(path: Path) -> bool:
    """Проверить, что каталог содержит сохранённый артефакт популяции."""
    return path.exists() and all((path / marker).exists() for marker in _ARTIFACT_MARKERS)


def available_sources(repo_root: Path) -> List[Tuple[str, bool, Path]]:
    """
    Перечислить источники и их готовность (есть ли собранный артефакт на диске).

    :param repo_root: корень репозитория
    :return: список кортежей ``(имя, готов, путь)``
    """
    return [(name, _is_artifact(dataset_dir(name, repo_root)), dataset_dir(name, repo_root))
            for name in DATASET_SOURCES]


def _ensure_synthetic(repo_root: Path, config: ExperimentConfig) -> Path:
    """Собрать синтетический артефакт, если его ещё нет, и вернуть путь."""
    target = dataset_dir("synthetic", repo_root)
    if not _is_artifact(target):
        from liquefaction_ai.data.synthetic import generate_population  # ленивый импорт

        population = generate_population(config)
        save_population_artifact(target, population, config)
    return target


def materialize_dataset(source: str, repo_root: Path, config: ExperimentConfig) -> Path:
    """
    Сделать выбранный источник активным датасетом пайплайна.

    Синтетика при отсутствии генерируется на месте; реальные источники должны быть
    предварительно собраны ноутбуками ``1_1_3`` / ``1_1_4``. Артефакт источника копируется
    в канонический каталог ``data/demo_run``.

    :param source: имя источника (ключ :data:`DATASET_SOURCES`)
    :param repo_root: корень репозитория
    :param config: конфигурация эксперимента (для генерации синтетики при необходимости)
    :return: путь к каноническому каталогу ``data/demo_run``
    """
    repo_root = Path(repo_root)
    if source == "synthetic":
        src = _ensure_synthetic(repo_root, config)
    else:
        src = dataset_dir(source, repo_root)
        if not _is_artifact(src):
            raise FileNotFoundError(
                f"Артефакт источника '{source}' не найден в {src}. Сначала соберите его "
                f"ноутбуком 1_1_3 (real_objects)."
            )

    # Проштамповать в config.json артефакта-источника его настоящее имя источника
    _stamp_dataset_source(src, source)

    canonical = repo_root / CANONICAL_DIR
    if canonical.resolve() != src.resolve():
        # Пофайловое перезаписывание (не rmtree+copytree): надёжнее, когда каталог-приёмник
        # лежит на ФС без права удаления (примонтированные папки), и сохраняет каталог на месте.
        canonical.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, canonical / item.name)
    # Канонический каталог тоже помечаем активным источником
    _stamp_dataset_source(canonical, source)
    return canonical


def _stamp_dataset_source(artifact_dir: Path, source: str) -> None:
    """Проставить ``dataset_source`` в ``config.json`` артефакта (метаданные источника)."""
    import json

    cfg_path = Path(artifact_dir) / "config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("dataset_source") != source:
            cfg["dataset_source"] = source
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except (ValueError, OSError):
        pass


def load_active_population(repo_root: Path):
    """
    Загрузить популяцию из канонического каталога ``data/demo_run``.

    :param repo_root: корень репозитория
    :return: кортеж ``(population, config)`` из :func:`load_population_artifact`
    """
    return load_population_artifact(Path(repo_root) / CANONICAL_DIR)
