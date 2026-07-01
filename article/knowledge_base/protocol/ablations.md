---
tags: [protocol, ablations, P1]
---
# Абляции (1:1 к заявленным вкладам)

Каждая абляция отключает ровно тот компонент, который заявлен как вклад. На grouped split + multiseed; свести в Table 3 с mean±CI и **разбивкой по 3 состояниям** (liq/stab/nostab).

## Уже есть (notebook 3_2)
- w/o flow · w/o ODE · NeuralODE w/o physics.

## Добавить
1. **w/o variance-scaling calibration** → Coverage@90 + ECE до/после (headline coverage = empirical site-held-out conformal, отдельно). Связь: [[../literature/romano-2019-cqr]].
2. **Flow-posterior vs честный Gaussian-posterior над θ** (не `use_flow` on/off). Доказать, что flow не overkill. Связь: [[../literature/rezende-mohamed-2015]].
3. **w/o monotonicity projection** (`monotone_clip`) → рост Physics_Violation_Rate, trade-off с RMSE.
4. **w/o discriminative risk head / soft-AUC** (`risk_clf`, `prior_gate`) → AUROC/AUPRC.
5. **Censoring/Tobit для N_liq** (with/without одностороннего censored loss). Связь: [[../literature/cox-1972]], [[../literature/nafday-2010]].
6. **Prefix-length sensitivity** (10/20/30/50%) → AUROC и N_liq vs префикс. Абляция + killer-фигура early-warning.
7. **Robustness к пропускам Vs/grainsize** (Vs только 16.7%, grainsize ~55%) → стресс на imputation.

Связано: [[evaluation-protocol]] · [[metrics]]

## ✅ Статус: РЕАЛИЗОВАНО (ноутбук **3_6** / `evaluation.ablation_study`)
Флаги в `models/dpi_flow.py` + раннер на объектном фолде (как P0), метрики **по 3 состояниям**.

| Абляция (рек.) | Как реализовано | Вариант в раннере |
|---|---|---|
| w/o variance-scaling calibration | не вызываем `fit_interval_scale` (calib_log_scale=0); empirical held-out coverage — отдельно, без formal conformal guarantee | `wo_varscale` |
| test-time θ-доводка 1/2 шага | `calibration_steps=1` / `=2` — эвристика, НЕ входит в нормированную плотность потока (headline = 0 шагов) | `calib_steps_1` / `calib_steps_2` |
| Flow vs честный Gaussian-posterior | `use_flow=False` | `gaussian_posterior` |
| w/o monotonicity projection | `use_monotone_clip=False` (→ bounded-clamp) | `wo_monotone` |
| black-box без ODE и без cummax | `use_analytical_layer=False, use_monotone_clip=False` | `blackbox_raw` |
| w/o discriminative risk / soft-AUC | `use_discriminative_risk=False` | `wo_risk_softauc` |
| Censoring/Tobit N_liq | `use_censored_nliq=False` (→ обычный MSE) | `wo_censored_nliq` |
| (структурная) w/o ODE | `use_analytical_layer=False` | `wo_ode` |
| Robustness к пропускам Vs | обнуление `V_s,Vs1` (=среднее) | `miss_vs` |
| Robustness к пропускам grainsize | обнуление `D_r,I_p,fines_content,clay_fraction,log10_Cu` | `miss_grainsize` |
| no-prefix / no-aux стресс | зануление префикса / derived g_obs | `no_prefix` / `no_aux` |
| Prefix-length sensitivity (п.6) | ре-материализация с разным `config.prefix_len`, затем `--only full --tag prefixK` | prefix-свип |

**A/B flow vs gaussian** (`evaluation.ab_test`): скоры на СТРОГО post-prefix continuation, кластер bootstrap по `site_id`, метрики mixture-NLL/CRPS + `Cov90_abs_miscal` (\|cov90−0.90\|); multi-fold через `ab_flow_vs_gaussian_pooled`.

Запуск: ноутбук **3_6** (агрегация в нём же → `ablations_summary.csv`). `QUICK=True` — дымовой тест. Фигуры — `ablation_bars` (3_6). См. [[p0-findings]].
