---
tags: [formulation, contributions]
---
# Тезис и вклады

Связано: [[positioning-and-title]] · [[../tracks/AAAI-27]]

## Тезис (one-sentence)
> We introduce a physics-structured probabilistic model that infers physically constrained latent liquefaction parameters from soil properties and an early PPR prefix, rolls them through a differentiable analytical CRR/damage/PPR layer (with an optional conditional flow for posterior flexibility), and produces calibrated forecasts of liquefaction onset, PPR continuation, risk, and physically feasible CRR/PPR trajectories under object-held-out evaluation.

## Contribution bullets (AAAI-стиль)
1. **Prefix-conditioned onset-forecasting formulation** для cyclic liquefaction tests: ранний PPR-префикс + soil descriptors → future PPR, risk, censored N_liq.
2. **Analytical differentiable liquefaction layer над физически ограниченными latent θ** — ядро вклада: amortized inference θ → дифференцируемый CRR/damage/PPR ODE-слой, гарантирующий feasible monotone accumulation. Conditional coupling (RealNVP) flow — **опциональный компонент** для гибкости posterior; в абляции его вклад скромный и зависит от датасета (gaussian posterior сопоставим/лучше на post-prefix RMSE/CRPS на текущем наборе), поэтому headline-вклад — физ-структура и amortized identification, а не сам flow.
3. **Censored onset objective:** liquefied exact; каждый landmark-eligible non-liq right-censored на фактическом last_obs. Stabilized/unfinished — отдельные физические regime-маски, не censoring-маски.
4. **Site-held-out benchmark:** архив ~1000 опытов / ~20 площадок; строгий landmark risk set (точные числа — в регенерируемом cohort-manifest) со strong baselines, группировка по `site_id`.
5. **Transparent uncertainty/physics evaluation:** post-prefix RMSE, censored N_liq, calibration, physics violations, CRR recovery (с N_CRR_test/N_CRR_objects).

## Короткий вариант contribution (для abstract)
> We propose DPI-Flow, a probabilistic physics-structured framework for prefix-conditioned liquefaction forecasting. The model infers constrained latent parameters, applies a conditional coupling (RealNVP) flow and prefix calibration, and integrates a differentiable CRR/damage/PPR ODE layer to predict PPR(N), liquefaction risk and censored N_liq under site-held-out validation.
