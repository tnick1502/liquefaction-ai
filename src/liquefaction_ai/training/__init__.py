"""
Подпакет обучения.

Содержит функции потерь (``losses``), универсальный цикл обучения (``loop``),
совместимый со всеми архитектурами пакета через единый контракт ``compute_loss``, и
помощники сохранения/загрузки обученных моделей с гиперпараметрами (``persistence``).
"""

from liquefaction_ai.training.losses import (
    clone_state_dict,
    gaussian_nll,
    masked_mae,
    masked_mean,
    masked_mse,
)
from liquefaction_ai.training.loop import evaluate_epoch_metrics, train_model
from liquefaction_ai.training.persistence import (
    load_model_metadata,
    load_weights_into,
    read_hyperparams,
    save_trained_model,
)
from liquefaction_ai.training.search import grid_search, iter_param_grid, write_hyperparams

__all__ = [
    "train_model",
    "evaluate_epoch_metrics",
    "gaussian_nll",
    "masked_mean",
    "masked_mse",
    "masked_mae",
    "clone_state_dict",
    "save_trained_model",
    "load_model_metadata",
    "load_weights_into",
    "read_hyperparams",
    "grid_search",
    "iter_param_grid",
    "write_hyperparams",
]
