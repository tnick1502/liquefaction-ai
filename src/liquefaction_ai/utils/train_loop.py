from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.utils.splits import iterate_minibatches
from liquefaction_ai.utils.losses import clone_state_dict


def train_model(
    model: nn.Module,
    train_split: Dict[str, object],
    val_split: Dict[str, object],
    epochs: int,
    model_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[nn.Module, pd.DataFrame]:
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = clone_state_dict(model)
    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
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
        val_losses = []
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
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = clone_state_dict(model)
        print(f"[{model_name}] эпоха {epoch:02d} | обучение={train_loss:.4f} | валидация={val_loss:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
