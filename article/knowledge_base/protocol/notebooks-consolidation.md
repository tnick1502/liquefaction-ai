---
tags: [protocol, refactor, notebooks, audit]
date: 2026-06-28
---
# Консолидация: скрипты → ноутбуки (аудит «ничего не потеряно»)

Все корневые `.py` удалены; логика перенесена в библиотечные модули `src/liquefaction_ai/evaluation/`
(+ `data/consistency.py`) и в тонкие ноутбуки `notebooks/3_evaluations/3_4..3_8`.

## Карта переноса
| Удалённый скрипт | Куда перенесено | Статус |
|---|---|---|
| run_cv.py | `evaluation/cross_validation.py` + NB **3_4** | ✓ полный перенос |
| run_significance.py | `evaluation/significance.py` + NB **3_5** | ✓ (валидировано на синтетике) |
| run_ablations.py | `evaluation/ablation_study.py` + NB **3_6** | ✓ полный перенос |
| run_p2_figs.py | `evaluation/publication.py` + NB **3_7** | ✓ (валидировано на синтетике) |
| run_consistency.py | `data/consistency.py` + NB **3_8** (+ тест) | ✓ полный перенос |
| run_crr_check.py / run_verify.py | покрыто `check_artifact_consistency` (Vs/plaxis/CRR/NaN) в NB **3_8** | ✓ объединено |
| run_p3_sensitivity.py | NB **3_8** (порт main) | ✓ полный перенос |
| run_multiseed.py | заменено объектным CV в NB **3_4** (grouped 5-fold + LOO) | ✓ улучшено (P0: grouped > random) |
| run_p1.py / run_p1b.py | финальные конфиги уже зашиты в модели (variance-scaling `fit_interval_scale`, censored loss) | ✓ superseded |
| run_ens.py | класс `models/ensemble.py::EnsembleModel` сохранён (опциональная capability) | ⚠ runner снят, класс остался |
| run_all.py / run_nbexec.py | заменено ручным порядком запуска ноутбуков (см. README) | ✓ оркестрация |
| incr_train / retrain_all / evt_retrain / dpi_evt_retrain | обучение — в NB **2_1..2_4** (best-of-seeds + лоссы уже там) | ✓ training в ноутбуках |

**Единственный нюанс:** deep-ensemble *runner* удалён, но класс `EnsembleModel` остался в
`src/liquefaction_ai/models/ensemble.py` — при необходимости ансамбль собирается из него (1 ячейка).

## Порядок перезапуска (локально)
1. **Данные:** `data/prepare_dataset.ipynb` — сборка единственного артефакта `data/dataset`
   (sites/ → dataset либо синтетика) **с фиксом утечки префикса** (`config.prefix_strict_preonset=True`).
   Это критично: иначе AUROC≈1.0 — артефакт утечки. Далее `1_data_analysis/1_1..1_3` только читают артефакт.
2. **Обучение:** `2_model_training/2_1` (бейзлайны) → `2_2` DPI-Flow → `2_3` EVT-NeuralSSM → `2_4` DPI-EVT.
3. **Базовая оценка:** `3_evaluations/3_1` (лидерборд) → `3_2` (старые ablations/OOD) → `3_3` (кейсы).
4. **Reviewer-response (P0–P2):** `3_4` object-CV+CI → `3_5` significance → `3_6` ablations → `3_7` figures → `3_8` consistency+P³-sensitivity.

Связано: [[p0-findings]] · [[ablations]] · [[metrics]] · [[evaluation-protocol]]
