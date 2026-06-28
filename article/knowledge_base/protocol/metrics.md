---
tags: [protocol, metrics, P2]
---
# Метрики (фокус, не количество)

Связано: [[evaluation-protocol]] · [[../literature/guo-2017-calibration]] · [[../literature/davis-goadrich-2006]] · [[../literature/romano-2019-cqr]]

## Onset (классификация)
- **AUPRC — primary** (дисбаланс 640/453), AUROC — вторично. Обоснование: [[../literature/davis-goadrich-2006]], [[../literature/fawcett-2006]].
- **Brier** ([[../literature/brier-1950]]).
- **Калибровка: ECE + reliability diagram (фигура)** ([[../literature/guo-2017-calibration]]). **Обязательно на grouped split** — OOD-калибровка обычно ломается; если держится → headline.
- **Lead-time / timeliness:** за сколько циклов до фактического onset поднимается риск → early-warning метрика.

## N_liq (event-time, цензура)
- Headline считать **только на нецензурированных** (или censored-aware), явно оговорив. MAE по right-censored атакуем. Обоснование цензуры: [[../literature/nafday-2010]], [[../literature/cox-1972]].
- Метрика: **N_liq_logMAE** (текущая лучшая у DPI-Flow 0.164).

## Trajectory (вероятностная)
- Вперёд **CRPS + NLL** (proper scoring), RMSE — diagnostic.
- **PICP / MPIW** (coverage + interval width) вместе. Conformal: [[../literature/romano-2019-cqr]].
- **Per-state (3-regime)** primary; `Traj_RMSE_worst` — robustness-история; pooled — вторично.

## CRR (уникальное)
- CRR_RMSE — **secondary**. Caveat: N_CRR_objects=1 в held-out test → claim осторожный (см. [[../tracks/AAAI-27]] limitations).
