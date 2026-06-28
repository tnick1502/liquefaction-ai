---
tags: [formulation, positioning]
---
# Позиционирование и title

Связано: [[../tracks/AAAI-27]] · [[thesis-and-contributions]]

## Главный принцип
Продавать **не** «нейросеть для разжижения», а **новый AI-метод** физически ограниченного вероятностного прогнозирования траектории и времени события в инженерной динамической системе. Геотехника = domain-critical benchmark, не единственная новизна.

## Track
**Main Technical Track.** AI for Social Impact — только запасной (если акцент на риск инфраструктуры/open data/общественную значимость).

## Позиционирующая фраза
> DPI-Flow is a physics-structured probabilistic latent-ODE/flow architecture for prefix-conditioned event-time forecasting under censoring and site-level distribution shift.

## Title (рекомендация)
`Physics-Constrained Conditional Flows for Liquefaction Onset Forecasting`
- **«Calibrated» в title — рано.** Coverage@90=0.975 = conservative, не идеально calibrated. Calibration → раздел/результат, не title-claim.
- Слова, которые должны быть в title/abstract: *physics-structured, probabilistic, ODE, event-time, prefix-conditioned, site-held-out.* «liquefaction» — как прикладной benchmark.

## Главный claim (честная формулировка)
NSF имеет лучший raw RMSE (0.1029), но Physics_Violation_Rate=0.917. Поэтому:
> **best physically admissible, onset-aware probabilistic forecaster under object-held-out evaluation** (НЕ «best RMSE overall»).

## Keywords
Machine Learning; Scientific ML; Physics-informed ML; Neural Differential Equations; Uncertainty Quantification; Calibration; Robustness/Generalization; Time-Series Forecasting; Event-Time Prediction; Survival Analysis; Tabular Learning; AI for Engineering.
