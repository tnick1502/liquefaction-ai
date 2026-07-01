"""
Сохранение и загрузка обученных моделей с гиперпараметрами.

Каждая модель сохраняется в собственный каталог ``models/<имя_модели>/`` тремя файлами:
- ``weights.pt`` — веса модели (``state_dict``);
- ``hyperparams.json`` — тип модели, аргументы конструктора и гиперпараметры обучения;
- ``history.parquet`` — история обучения (кривые train/val по эпохам).

Это позволяет ноутбукам серии 03 загружать модели для оценки, восстанавливая их по
сохранённым гиперпараметрам без дублирования кода конфигурации.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

__all__ = ["save_trained_model", "load_model_metadata", "load_weights_into", "read_hyperparams"]


def read_hyperparams(models_root: str | Path, name: str) -> Dict[str, Any]:
    """
    Прочитать гиперпараметры модели (``hyperparams.json``) без истории обучения.

    Используется финальным обучением: гиперпараметры, выбранные grid search, читаются
    из каталога модели до старта обучения.

    :param models_root: корневой каталог моделей
    :param name: имя модели (имя подкаталога)
    :return: словарь гиперпараметров
    """
    path = Path(models_root) / name / "hyperparams.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_trained_model(
    model: nn.Module,
    models_root: str | Path,
    name: str,
    hyperparams: Dict[str, Any],
    history: pd.DataFrame,
) -> Path:
    """
    Сохранить обученную модель в собственный каталог с гиперпараметрами и историей.

    :param model: обученная модель PyTorch
    :param models_root: корневой каталог моделей (например, ``<корень>/models``)
    :param name: имя модели (становится именем подкаталога)
    :param hyperparams: словарь гиперпараметров; рекомендуется включать ключи
                        ``model_type`` (имя класса) и ``model_kwargs`` (аргументы конструктора)
    :param history: история обучения (DataFrame с колонками ``epoch``/``train_loss``/``val_loss``)
    :return: путь к каталогу сохранённой модели
    """
    out_dir = Path(models_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "hyperparams.json").write_text(json.dumps(hyperparams, indent=2, ensure_ascii=False), encoding="utf-8")
    history.to_parquet(out_dir / "history.parquet")
    return out_dir


def load_model_metadata(models_root: str | Path, name: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Загрузить гиперпараметры и историю обучения сохранённой модели.

    :param models_root: корневой каталог моделей
    :param name: имя модели (имя подкаталога)
    :return: кортеж (словарь гиперпараметров, история обучения DataFrame)
    """
    model_dir = Path(models_root) / name
    hyperparams = json.loads((model_dir / "hyperparams.json").read_text(encoding="utf-8"))
    history = pd.read_parquet(model_dir / "history.parquet")
    return hyperparams, history


def load_weights_into(model: nn.Module, models_root: str | Path, name: str, device: torch.device) -> nn.Module:
    """
    Загрузить сохранённые веса в уже созданную модель.

    :param model: экземпляр модели с архитектурой, совпадающей с сохранённой
    :param models_root: корневой каталог моделей
    :param name: имя модели (имя подкаталога)
    :param device: устройство, на которое загружаются веса
    :return: та же модель с загруженными весами (в режиме eval)
    """
    weights_path = Path(models_root) / name / "weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
