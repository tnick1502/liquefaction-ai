---
tags: [protocol, final-run, training, grid-search]
date: 2026-06-28
---
# Настройки финального (публикационного) прогона

Связано: [[notebooks-consolidation]] · [[reviewer-remarks-fixes]] · [[evaluation-protocol]]

## Серьёзное обучение (config = единый источник истины)
`src/liquefaction_ai/config.py`:
- `publication_physics_epochs = 200`, `publication_baseline_epochs = 120` — **потолок** эпох;
  реально обучение останавливает **ранняя остановка** по best-val.
- `early_stopping_patience = 25`, `early_stopping_min_delta = 1e-4` — early stopping встроен в
  `train_model` (восстанавливает лучший по val чекпоинт).
- Косинусный LR с прогревом — **для всех torch-моделей** (в CV `_train_one` теперь `scheduler="cosine"`,
  не только для физических).
- `grid_search_epochs = 20` — серьёзный грид-сёрч (вместо 1–2 эпох).
- demo-режим (`baseline_epochs=4`, `physics_epochs=6`) остаётся только для `QUICK=True`/`--quick`.

## Грид-сёрч (ноутбуки 2_x, на уровне)
- **2_1** baselines: `search_epochs=config.grid_search_epochs`; финальное обучение —
  `publication_*` эпохи (PINN — physics, остальные — baseline).
- **2_2** DPI-Flow: grid `hidden_dim∈{128,160,192} × use_traj_residual∈{T,F}` при `calibration_steps=0`
  (headline = нормированная плотность потока; θ-доводка 1/2 — только абляция),
  `search_epochs=config.grid_search_epochs`, финал `publication_physics_epochs`.
- **2_3** EVT-NeuralSSM: grid `hidden_dim∈{96,128,160}`, аналогично.
- **2_4** DPI-EVT: финал `publication_physics_epochs` (наследует tuned `hidden_dim` от EVT).
Лучшие гиперпараметры пишутся в `models/<name>/hyperparams.json` → их читает и финальное обучение,
и кросс-валидация (3_4) при конструировании моделей.

## Стоимость и тумблеры (важно)
Полный финал тяжёлый: 3_4 = 9 моделей × 5 фолдов × `N_REPEATS` повторов × (до 200 эпох с ES).
Тумблеры под железо/время:
- `N_REPEATS` в 3_4 (по умолчанию 3) — снизить до 1–2 при нехватке времени;
- `FOLDS` в 3_6 (по умолчанию [0,1,2]) — ключевые абляции;
- `QUICK=True` — дымовой прогон на demo-эпохах (НЕ для headline);
- `RUN_LOO` в 3_4 — LOO-20 как secondary (дорого).

## Порядок финального прогона
1. `1_x` — перематериализация (strict prefix).
2. **`2_1`→`2_2`→`2_3`→`2_4`** — серьёзный грид-сёрч + публикационное обучение, сохранение моделей/hyperparams.
3. `3_1`→`3_3` — лидерборд/база.
4. `3_4`→`3_5`→`3_6`→`3_7`→`3_8` — CV+CI, значимость, абляции, фигуры, consistency/P³.
