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
# (ведомости digitrock, сборка data/prepare_dataset.ipynb). Менять порог нужно ТОЛЬКО здесь.
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
    :param baseline_epochs: число эпох для быстрых demo-прогонов базовых моделей (MLP/GRU/TCN)
    :param physics_epochs: число эпох для быстрых demo-прогонов физически-структурированных моделей
    :param ablation_epochs: число эпох в абляционных и OOD-экспериментах
    :param learning_rate: скорость обучения оптимизатора AdamW
    :param weight_decay: коэффициент L2-регуляризации оптимизатора AdamW
    :param mc_samples_eval: число Monte-Carlo сэмплов при вероятностной оценке
    :param export_figures: сохранять ли рисунки на диск
    :param figure_dir: каталог для экспорта рисунков
    :param max_csr_clip: верхняя отсечка значений CSR(N) в генераторе
    :param max_cycle_reference: практический горизонт N_liq для логарифмической нормировки и
        цензурированных метрик; после этого горизонта разжижение считается практически
        ненаступившим в инженерном смысле
    :param risk_threshold: порог классификации риска разжижения по умолчанию
    :param measured_crr_fraction: доля грунтов с «измеренной» кривой CRR(N) (имитация серии из
        6 образцов); такие кривые используются как опциональная наблюдаемая супервизия границы CRR
    :param dataset_source: активный источник данных пайплайна — ``synthetic`` или
        ``real_objects`` (выбирается в ``data/prepare_dataset.ipynb`` через ``LIQ_DATASET``)
    :param liq_threshold: канонический порог события разжижения по ru=PPR (см. :data:`LIQ_THRESHOLD`);
        единый для метки, N_liq, наблюдаемого триггера g_obs и пересечения в моделях
    :param min_nonliq_complete_cycles: минимальная длительность неразжижившегося опыта, после
        которой плоский хвост PPR можно считать наблюдаемой стабилизацией; если циклов меньше,
        терминал N_liq считается неоценимым даже при визуально плоском хвосте
    :param publication_baseline_epochs: рекомендуемый минимум эпох для отчётных, не-demo прогонов
        базовых моделей; нужен, чтобы README/таблицы не выдавали быстрый sanity-check за финальное
        сравнение архитектур
    :param publication_physics_epochs: рекомендуемый минимум эпох для отчётных прогонов
        физически-структурированных моделей
    :param early_stopping_patience: число эпох без улучшения validation loss, после которого
        обучение останавливается досрочно; None/0 отключает остановку
    :param early_stopping_min_delta: минимальное улучшение validation loss, считающееся прогрессом
    :param use_observed_aux_loss: включать auxiliary supervision, выводимую из полной наблюдаемой
        PPR-кривой (g_obs/risk_proxy/CRR_obs), при обучении. Для стресс-теста
        no-derived-threshold-auxiliary установите False.
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
    baseline_epochs: int = 4               # demo/дымовой режим (быстро)
    physics_epochs: int = 6                # demo/дымовой режим (быстро)
    # --- ФИНАЛЬНЫЙ (публикационный) режим: много эпох + ранняя остановка по best-val ---
    publication_baseline_epochs: int = 120   # потолок; реально остановит early stopping
    publication_physics_epochs: int = 200    # потолок; реально остановит early stopping
    grid_search_epochs: int = 20             # серьёзный грид-сёрч (не 1–2 эпохи)
    early_stopping_patience: int = 25        # терпеливее для длинного обучения
    early_stopping_min_delta: float = 1e-4
    ablation_epochs: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    mc_samples_eval: int = 8
    export_figures: bool = False
    figure_dir: str = "reports/liquefaction_demo_figures"
    max_csr_clip: float = 0.65
    max_cycle_reference: float = 3_000.0
    risk_threshold: float = 0.5
    measured_crr_fraction: float = 0.25
    dataset_source: str = "synthetic"
    liq_threshold: float = LIQ_THRESHOLD   # канонический порог события разжижения ru=PPR (см. LIQ_THRESHOLD)
    min_nonliq_complete_cycles: float = 500.0
    use_observed_aux_loss: bool = True
    group_split_by_object: bool = True     # Основной протокол: leakage-free разбиение по объекту/площадке
    #                                        (ни один объект не попадает одновременно в train/val/test)
    # --- Анти-утечка префикса (P0-c). Наблюдаемый префикс ОБРЕЗАЕТСЯ строго до onset разжижения,
    #     иначе на быстрых опытах вход уже содержит само событие (≈24% разжижающихся) и AUROC≈1.0
    #     становится артефактом утечки метки через вход. См. data.real_adapter.strict_pre_onset_prefix_mask.
    prefix_strict_preonset: bool = True     # обрезать префикс строго до onset (рекоменд. протокол)
    prefix_onset_threshold: float = LIQ_THRESHOLD   # порог ru, определяющий onset (тот же, что у события)
    prefix_onset_margin: int = 1            # доп. буфер шагов: последний шаг префикса < onset_idx − margin
    prefix_min_len: int = 3                 # минимальная длина префикса, НО только если не пересекает onset
    # --- Протокол префикса. "preonset" — основной (анти-утечка, длина зависит от onset → outcome-dependent).
    #     "fixed_k" — отдельный leakage-free протокол: ФИКСИРОВАННОЕ окно первых prefix_fixed_k шагов для
    #     ВСЕХ опытов, не зависящее от исхода; K выбран малым, чтобы разжижение почти не попадало в окно.
    # "landmark" (рекомендуемый primary, leakage-free onset forecasting): префикс = наблюдения до
    #   ФИЗИЧЕСКОГО landmark-цикла N₀; risk set — только опыты, не разжижившиеся до N₀ (см. splits).
    # "fixed_k" — фикс. окно первых K шагов сетки. "preonset" — обрезка до onset (outcome-dependent).
    prefix_mode: str = "landmark"           # "landmark" | "fixed_k" | "preonset"
    prefix_fixed_k: int = 6                 # длина фиксированного префикса (шагов) для prefix_mode="fixed_k"
    prefix_landmark_cycles: float = 10.0    # физический landmark N₀ (циклы) для prefix_mode="landmark".
    # N₀=10 (а не 20): N₀=20 целиком выбрасывал 6 из 19 площадок (опыты <20 циклов — короткие сейсмо-
    # протоколы), что подрывало site-held-out claim. При N₀=10 сохраняются ~19/19 сайтов ценой более
    # короткого префикса. Выбор задокументирован sensitivity-таблицей (cohort × n_sites × длина префикса).


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
