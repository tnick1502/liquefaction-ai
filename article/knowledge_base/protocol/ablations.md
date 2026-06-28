---
tags: [protocol, ablations, P1]
---
# Абляции (1:1 к заявленным вкладам)

Каждая абляция отключает ровно тот компонент, который заявлен как вклад. На grouped split + multiseed; свести в Table 3 с mean±CI и **разбивкой по 3 состояниям** (liq/stab/nostab).

## Уже есть (notebook 3_2)
- w/o flow · w/o ODE · NeuralODE w/o physics.

## Добавить
1. **w/o conformal calibration** → Coverage@90 + ECE до/после. Опора claim «calibrated». Связь: [[../literature/romano-2019-cqr]].
2. **Flow-posterior vs честный Gaussian-posterior над θ** (не `use_flow` on/off). Доказать, что flow не overkill. Связь: [[../literature/rezende-mohamed-2015]].
3. **w/o monotonicity projection** (`monotone_clip`) → рост Physics_Violation_Rate, trade-off с RMSE.
4. **w/o discriminative risk head / soft-AUC** (`risk_clf`, `prior_gate`) → AUROC/AUPRC.
5. **Censoring/Tobit для N_liq** (with/without одностороннего censored loss). Связь: [[../literature/cox-1972]], [[../literature/nafday-2010]].
6. **Prefix-length sensitivity** (10/20/30/50%) → AUROC и N_liq vs префикс. Абляция + killer-фигура early-warning.
7. **Robustness к пропускам Vs/grainsize** (Vs только 16.7%, grainsize ~55%) → стресс на imputation.

Связано: [[evaluation-protocol]] · [[metrics]]
