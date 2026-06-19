"""
Конфигурация эксперимента и глобальные настройки воспроизводимости.

Модуль задаёт единый источник истины для всех гиперпараметров пайплайна
(генерация данных, разбиение на выборки, обучение моделей, оценка метрик),
а также утилиту фиксации случайных зёрен и единую палитру визуализаций.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "ExperimentConfig",
    "get_default_config",
    "set_global_seed",
    "DEMO_PALETTE",
    "LIQ_THRESHOLD",
]

# Единый канонический порог события разжижения по поровому давлению ru = PPR.
# Один и тот же порог описывает ВСЕ компоненты события: бинарную метку разжижения, число
# циклов N_liq, наблюдаемый триггер g_obs (auxiliary supervision) и момент пересечения в
# моделях. Значение ru ≥ 0.95 соответствует определению разжижения в исходных данных
# (ведомости digitrock, загрузчик 1_1_3). Менять порог нужно ТОЛЬКО здесь.
LIQ_THRESHOLD: float = 0.95


@dataclass
class ExperimentConfig:
    """
    Контейнер гиперпараметров эксперимента по моделированию разжижения.

    Один экземпляр конфигурации проходит через весь пайплайн: генератор
    синтетической популяции, стратифицированное разбиение benchmark-подмножества,
    цикл обучения и блок оценки. Это гарантирует согласованность параметров между
    ноутбуками серий 01–04.

    :param seed: глобальное случайное зерно для numpy/torch/random
    :param n_scenarios: размер полной синтетической популяции сценариев
    :param benchmark_subset: размер стратифицированного benchmark-подмножества для обучения
    :param ablation_subset: размер подмножества для быстрых абляционных экспериментов
    :param seq_len: длина временной последовательности (число узлов сетки по циклам N)
    :param prefix_len: длина наблюдаемого префикса траектории PPR
    :param benchmark_train_fraction: доля обучающей выборки внутри benchmark
    :param benchmark_val_fraction: доля валидационной выборки внутри benchmark
    :param batch_size: размер мини-батча при обучении
    :param baseline_epochs: число эпох для базовых моделей (MLP/GRU/TCN)
    :param physics_epochs: число эпох для физически-структурированных моделей (DPI-Flow/EVT-NeuralSSM)
    :param ablation_epochs: число эпох в абляционных и OOD-экспериментах
    :param learning_rate: скорость обучения оптимизатора AdamW
    :param weight_decay: коэффициент L2-регуляризации оптимизатора AdamW
    :param mc_samples_eval: число Monte-Carlo сэмплов при вероятностной оценке
    :param export_figures: сохранять ли рисунки на диск
    :param figure_dir: каталог для экспорта рисунков
    :param max_csr_clip: верхняя отсечка значений CSR(N) в генераторе
    :param max_cycle_reference: опорное максимальное число циклов для логарифмической нормировки N_liq
    :param risk_threshold: порог классификации риска разжижения по умолчанию
    :param measured_crr_fraction: доля грунтов с «измеренной» кривой CRR(N) (имитация серии из
        6 образцов); такие кривые используются как опциональная наблюдаемая супервизия границы CRR
    :param dataset_source: активный источник данных пайплайна — ``synthetic`` или
        ``real_objects`` (см. :mod:`liquefaction_ai.data.dataset_source`)
    :param liq_threshold: канонический порог события разжижения по ru=PPR (см. :data:`LIQ_THRESHOLD`);
        единый для метки, N_liq, наблюдаемого триггера g_obs и пересечения в моделях
    """

    seed: int = 42
    n_scenarios: int = 24_000
    benchmark_subset: int = 8_000
    ablation_subset: int = 4_000
    seq_len: int = 72
    prefix_len: int = 12
    benchmark_train_fraction: float = 0.70
    benchmark_val_fraction: float = 0.15
    batch_size: int = 256
    baseline_epochs: int = 4
    physics_epochs: int = 6
    ablation_epochs: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    mc_samples_eval: int = 8
    export_figures: bool = False
    figure_dir: str = "reports/liquefaction_demo_figures"
    max_csr_clip: float = 0.65
    max_cycle_reference: float = 1_500.0
    risk_threshold: float = 0.5
    measured_crr_fraction: float = 0.25
    dataset_source: str = "synthetic"
    liq_threshold: float = LIQ_THRESHOLD   # канонический порог события разжижения ru=PPR (см. LIQ_THRESHOLD)
    group_split_by_object: bool = True     # Основной протокол: leakage-free разбиение по объекту/площадке
    #                                        (ни один объект не попадает одновременно в train/val/test)


def get_default_config() -> ExperimentConfig:
    """
    Получить конфигурацию эксперимента со значениями по умолчанию.

    :return: новый экземпляр :class:`ExperimentConfig` с дефолтными параметрами
    """
    return ExperimentConfig()


def set_global_seed(seed: int) -> None:
    """
    Зафиксировать случайные зёрна во всех используемых библиотеках.

    Делает прогон детерминированным: фиксируются генераторы ``random``, ``numpy``
    и ``torch`` (включая CUDA при наличии), а также отключается недетерминированный
    режим cuDNN. Это критично для воспроизводимости синтетических данных и обучения.

    :param seed: значение случайного зерна
    :return: None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Полный детерминизм на CPU: один поток + детерминированные алгоритмы torch.
    # Это устраняет дрейф метрик между прогонами (особенно у близких по качеству моделей).
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


DEMO_PALETTE = {
    "primary": "#0b6efd",
    "secondary": "#6610f2",
    "accent": "#d63384",
    "success": "#198754",
    "warning": "#fd7e14",
    "danger": "#dc3545",
    "dark": "#1f2937",
    "sand": "#c99a3d",
    "silt": "#8b9dc3",
}
"""Единая палитра цветов для визуализаций конференционного уровня."""
