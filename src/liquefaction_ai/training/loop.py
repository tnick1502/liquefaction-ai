"""
Универсальный цикл обучения моделей.

Один цикл подходит для всех архитектур пакета: они реализуют единый контракт —
метод ``compute_loss(batch)``, возвращающий словарь с ключом ``loss``. Цикл выполняет
обучение с AdamW и отсечением градиентов, валидацию каждой эпохи и раннее сохранение
лучшего по валидации состояния модели. Опционально на каждой эпохе вычисляются
валидационные метрики (AUROC, Brier, RMSE траектории) для построения «живых» кривых
обучения.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.data.splits import iterate_minibatches
from liquefaction_ai.training.losses import clone_state_dict

__all__ = ["train_model", "evaluate_epoch_metrics"]


def evaluate_epoch_metrics(
    model: nn.Module,
    split: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Вычислить лёгкий набор валидационных метрик для мониторинга обучения.

    Считаются AUROC и Brier классификации риска, а при наличии траекторной головы —
    RMSE траектории PPR на валидной части последовательности. Используется внутри
    цикла обучения для построения кривых метрик по эпохам.

    :param model: текущая модель
    :param split: валидационная выборка
    :param config: конфигурация эксперимента (размер батча)
    :param device: устройство инференса
    :return: словарь метрик (``val_auroc``, ``val_brier`` и, если применимо, ``val_traj_rmse``)
    """
    from liquefaction_ai.evaluation.metrics import collect_outputs, safe_binary_metrics

    outputs = collect_outputs(model, split, config, device)
    y_true = split["label"].cpu().numpy()
    auroc, _, brier = safe_binary_metrics(y_true, outputs["risk_prob"])
    metrics: Dict[str, float] = {"val_auroc": auroc, "val_brier": brier}

    if "traj_mean" in outputs:
        pred = outputs["traj_mean"]
        true = split["r_true"].cpu().numpy()
        mask = split["mask"].cpu().numpy()
        mse = float(np.sum(((pred - true) ** 2) * mask) / np.maximum(mask.sum(), 1.0))
        metrics["val_traj_rmse"] = float(np.sqrt(mse))
    return metrics


def train_model(
    model: nn.Module,
    train_split: Dict[str, object],
    val_split: Dict[str, object],
    epochs: int,
    model_name: str,
    config: ExperimentConfig,
    device: torch.device,
    track_metrics: bool = False,
    verbose: bool = True,
) -> Tuple[nn.Module, pd.DataFrame]:
    """
    Обучить модель и вернуть лучшее по валидации состояние и историю обучения.

    Каждую эпоху выполняется проход по обучающим мини-батчам с обратным
    распространением и отсечением нормы градиента (max_norm=1.0), затем оценивается
    средний loss на валидации. Состояние с наименьшим валидационным loss сохраняется и
    восстанавливается в конце (ранняя остановка по лучшему чекпоинту). При
    ``track_metrics=True`` дополнительно фиксируются валидационные метрики по эпохам.

    :param model: модель PyTorch с методом ``compute_loss(batch) -> {"loss": ...}``
    :param train_split: обучающая выборка (из ``prepare_benchmark_dataset``)
    :param val_split: валидационная выборка
    :param epochs: число эпох обучения
    :param model_name: отображаемое имя модели для логов
    :param config: конфигурация эксперимента (lr, weight_decay, batch_size, seed)
    :param device: устройство обучения
    :param track_metrics: фиксировать ли валидационные метрики (AUROC/Brier/RMSE) по эпохам
    :param verbose: печатать ли прогресс по эпохам
    :return: кортеж (обученная модель с лучшими весами, история обучения в виде DataFrame
             с колонками ``epoch``/``train_loss``/``val_loss`` и, при ``track_metrics``, метриками)
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = clone_state_dict(model)
    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in iterate_minibatches(
            train_split,
            config.batch_size,
            device,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.compute_loss(batch)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss_dict["loss"].detach().cpu()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in iterate_minibatches(
                val_split,
                config.batch_size,
                device,
                shuffle=False,
            ):
                loss_dict = model.compute_loss(batch)
                val_losses.append(float(loss_dict["loss"].detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        record: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if track_metrics:
            record.update(evaluate_epoch_metrics(model, val_split, config, device))
        history.append(record)

        if val_loss < best_val:
            best_val = val_loss
            best_state = clone_state_dict(model)
        if verbose:
            extra = ""
            if track_metrics:
                extra = f" | val_AUROC={record.get('val_auroc', float('nan')):.3f}"
                if "val_traj_rmse" in record:
                    extra += f" | val_RMSE={record['val_traj_rmse']:.4f}"
            print(f"[{model_name}] эпоха {epoch:02d} | обучение={train_loss:.4f} | валидация={val_loss:.4f}{extra}", flush=True)

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
