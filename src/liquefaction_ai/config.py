from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "ExperimentConfig",
    "get_default_config",
    "set_global_seed",
    "DEMO_PALETTE",
]


@dataclass
class ExperimentConfig:
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
    physics_epochs: int = 5
    ablation_epochs: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    mc_samples_eval: int = 8
    export_figures: bool = False
    figure_dir: str = "reports/liquefaction_demo_figures"
    max_csr_clip: float = 0.65
    max_cycle_reference: float = 1_500.0
    risk_threshold: float = 0.5


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
