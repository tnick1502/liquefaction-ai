---
tags: [track, neurips-workshop, secondary]
status: parallel
---
# Трек 2 — NeurIPS Workshop (topology / early-warning)

**Фокус:** более исследовательская/рискованная часть. **НЕ смешивать с AAAI.**

## Идея
Латентное пространство моделей и траектории скрытых состояний содержат **ранние признаки** перехода грунта к разжижению. DPI-Flow и EVT-NeuralSSM здесь — инструменты анализа латентной геометрии, режимов грунта и early-warning сигналов, а не главные модели.

## Содержание
- Topology / Mapper / UMAP над латентными θ.
- Early-warning signals (critical slowing down и т.п.) на траекториях скрытых состояний.
- Latent regime geometry (3 состояния: liq / stab / nostab — см. memory `liquefaction-regime-aware-p3`).

## Связи
- Метод-якоря: [[../literature/rezende-mohamed-2015]] · [[../literature/chen-2018-neuralode]]
- В AAAI это **только** маленькая interpretability-панель + future work: [[AAAI-27]]
