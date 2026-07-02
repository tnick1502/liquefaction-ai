"""
Пакет моделирования разжижения грунтов: DPI-Flow и EVT-NeuralSSM.

Структура пакета организована по доменам:
- ``config`` / ``constants`` — конфигурация эксперимента и таксономия грунтов/нагружений;
- ``data`` — генерация синтетической популяции, сплиты, сериализация, локализация;
- ``physics`` — аналитические модели CRR и PPR (теоретическая основа);
- ``models`` — нейросетевые архитектуры (базовые и физически-структурированные);
- ``training`` — функции потерь и универсальный цикл обучения;
- ``evaluation`` — метрики качества, агрегаты и эксперименты;
- ``viz`` — визуализация кривых (Plotly) и разведочного анализа (matplotlib).

Наиболее востребованные объекты реэкспортируются на верхний уровень пакета.
"""

from liquefaction_ai.accel import configure_performance, describe_device, resolve_device
from liquefaction_ai.config import DEMO_PALETTE, ExperimentConfig, get_default_config, set_global_seed
from liquefaction_ai.data import (
    generate_population,
    load_population_artifact,
    prepare_benchmark_dataset,
    save_population_artifact,
)
from liquefaction_ai.training import train_model

__all__ = [
    "DEMO_PALETTE",
    "ExperimentConfig",
    "get_default_config",
    "set_global_seed",
    "generate_population",
    "save_population_artifact",
    "load_population_artifact",
    "prepare_benchmark_dataset",
    "train_model",
    "resolve_device",
    "configure_performance",
    "describe_device",
]
