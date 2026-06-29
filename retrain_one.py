"""
Лёгкий переобучатель ОДНОЙ структурной модели под новую параметризацию потока.

Полные ноутбуки 2_2/2_4 (grid search + best-of-seeds + publication_physics_epochs=200) рассчитаны
на железо пользователя и не укладываются в ограничение песочницы. После смены аффинного потока на
настоящий conditional RealNVP старые веса несовместимы, поэтому этот скрипт переобучает модель по
сохранённым (из прошлого grid search) гиперпараметрам — единичный сид, заданное число эпох,
косинусный LR — и сохраняет рабочие веса, чтобы пайплайн оценки запускался end-to-end. Для
публикационных чисел запустите 2_2/2_4 целиком на своём железе.

    python retrain_one.py <dpi_flow|dpi_evt> <epochs>
"""
import os
import sys

os.environ.setdefault("LIQ_DATASET", "real_objects")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

from liquefaction_ai import (get_default_config, load_population_artifact,  # noqa: E402
                             prepare_benchmark_dataset, train_model)
from liquefaction_ai.config import set_global_seed  # noqa: E402
from liquefaction_ai.evaluation import fit_interval_scale  # noqa: E402
from liquefaction_ai.training import read_hyperparams, save_trained_model  # noqa: E402
from liquefaction_ai import models as M  # noqa: E402

name = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MODELS = os.path.join(REPO, "models")
cfg = get_default_config(); dev = torch.device("cpu")
set_global_seed(cfg.seed)
pop, cfg = load_population_artifact(os.path.join(REPO, "data", "demo_run"))
bench = prepare_benchmark_dataset(pop, cfg, dev)

hp = read_hyperparams(MODELS, name)
cls = getattr(M, hp["model_type"])
set_global_seed(cfg.seed)
model = cls(**hp["model_kwargs"]).to(dev)
n_params = sum(p.numel() for p in model.parameters())
model, history = train_model(model, bench["train"], bench["val"], epochs=epochs,
                             model_name=name, config=cfg, device=dev, verbose=False,
                             track_metrics=True, scheduler="cosine")
calib_scale = fit_interval_scale(model, bench["val"], cfg, dev, level=0.90)
save_trained_model(model, MODELS, name, {**hp, "epochs": epochs, "calib_scale": float(calib_scale),
                                         "note": "quick-retrain под новый RealNVP-поток; "
                                                 "для публикации перезапустить 2_2/2_4 целиком"},
                   history)
print(f"{name}: переобучено {epochs} эпох, params={n_params:,}, calib_scale={calib_scale:.3f} → {MODELS}/{name}")
