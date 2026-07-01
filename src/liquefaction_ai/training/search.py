"""
Подбор гиперпараметров перебором по сетке (grid search).

Перебирает комбинации гиперпараметров, кратко обучает модель на каждой комбинации,
оценивает на валидации по выбранной метрике и возвращает таблицу результатов и лучшую
комбинацию. Лучшие гиперпараметры сохраняются в каталог модели (``hyperparams.json``),
после чего финальное обучение читает этот файл и обучает модель «начисто».
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from liquefaction_ai.config import ExperimentConfig, set_global_seed
from liquefaction_ai.training.loop import train_model

__all__ = ["iter_param_grid", "grid_search", "write_hyperparams"]


def iter_param_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Развернуть сетку гиперпараметров в список всех комбинаций.

    :param grid: словарь «имя параметра → список значений»
    :return: список словарей-комбинаций (декартово произведение значений)
    """
    keys = list(grid.keys())
    combos = []
    for values in product(*[grid[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def grid_search(
    build_fn: Callable[[Dict[str, Any]], nn.Module],
    grid: Dict[str, List[Any]],
    train_split: Dict[str, object],
    val_split: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
    search_epochs: int = 2,
    score_metric: str = "Traj_RMSE",
    scheduler: str = "cosine",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Перебрать сетку гиперпараметров, оценить полный набор метрик и выбрать лучшую комбинацию.

    Для каждой комбинации модель строится фабрикой ``build_fn``, кратко обучается на
    ``search_epochs`` эпох и оценивается на валидации. В таблицу результатов записываются
    **все** метрики из каталога (``val_loss`` плюс траекторные и вероятностные метрики из
    ``compute_metrics``) — это богатая история перебора. Лучшая комбинация выбирается по
    метрике ``score_metric`` с учётом её направления (см. :data:`METRICS`).

    :param build_fn: фабрика модели по словарю гиперпараметров
    :param grid: сетка гиперпараметров (словарь списков значений)
    :param train_split: обучающая выборка
    :param val_split: валидационная выборка
    :param config: конфигурация эксперимента
    :param device: устройство
    :param search_epochs: число эпох краткого обучения на каждой комбинации
    :param score_metric: метрика отбора лучшей комбинации (любой ключ из каталога метрик)
    :return: кортеж (таблица результатов со всеми метриками по комбинациям, лучшая комбинация)
    """
    from liquefaction_ai.evaluation.metrics import collect_outputs, compute_metrics, rank_by_metric

    rows: List[Dict[str, Any]] = []
    for params in iter_param_grid(grid):
        set_global_seed(config.seed) # одинаковая инициализация кандидатов → стабильный отбор
        model = build_fn(params).to(device)
        model, history = train_model(
            model, train_split, val_split, epochs=search_epochs,
            model_name="grid-search", config=config, device=device, verbose=False, scheduler=scheduler,
        )
        outputs = collect_outputs(model, val_split, config, device)
        metrics, _ = compute_metrics("grid-search", outputs, val_split, config)
        metrics.pop("model", None)
        row: Dict[str, Any] = {**params, "val_loss": float(history["val_loss"].iloc[-1])}
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()})
        rows.append(row)

    results = rank_by_metric(pd.DataFrame(rows), score_metric)
    # Извлекаем значения из колонок (а не из строки), чтобы pandas не приводил int к float
    best_params: Dict[str, Any] = {}
    for key in grid.keys():
        value = results[key].iloc[0]
        best_params[key] = value.item() if hasattr(value, "item") else value
    return results, best_params


def write_hyperparams(models_root: str | Path, name: str, payload: Dict[str, Any]) -> Path:
    """
    Сохранить выбранные гиперпараметры в каталог модели (``hyperparams.json``).

    После этого финальное обучение читает этот файл и обучает модель «начисто».

    :param models_root: корневой каталог моделей
    :param name: имя модели (имя подкаталога)
    :param payload: словарь гиперпараметров (рекомендуется ключи ``model_type``,
                    ``display_name``, ``model_kwargs`` и блок ``search``)
    :return: путь к сохранённому файлу ``hyperparams.json``
    """
    out_dir = Path(models_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hyperparams.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
