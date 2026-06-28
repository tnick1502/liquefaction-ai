---
tags: [formulation, contributions]
---
# Тезис и вклады

Связано: [[positioning-and-title]] · [[../tracks/AAAI-27]]

## Тезис (one-sentence)
> We introduce a physics-constrained conditional flow that infers a distribution over latent liquefaction parameters from soil properties and an early PPR prefix, rolls them through a differentiable analytical layer, and produces forecasts of liquefaction onset, PPR continuation, risk, and physically feasible CRR/PPR trajectories under object-held-out evaluation.

## Contribution bullets (AAAI-стиль)
1. **Prefix-conditioned onset-forecasting formulation** для cyclic liquefaction tests: ранний PPR-префикс + soil descriptors → future PPR, risk, censored N_liq.
2. **Conditional parameter flow + analytical differentiable liquefaction layer:** flow выводит distribution over physical θ; ODE-layer гарантирует feasible monotone accumulation.
3. **Censored onset objective:** liquefied exact; stabilized non-liq right-censored; unfinished non-liq excluded from N_liq supervision, но kept для trajectory.
4. **Object-held-out benchmark** на 1093 реальных лабораторных опытах со strong baselines.
5. **Transparent uncertainty/physics evaluation:** post-prefix RMSE, censored N_liq, calibration, physics violations, CRR recovery (с N_CRR_test/N_CRR_objects).

## Короткий вариант contribution (для abstract)
> We propose DPI-Flow, a probabilistic physics-structured framework for prefix-conditioned liquefaction forecasting. The model infers constrained latent parameters, applies a conditional affine flow and prefix calibration, and integrates a differentiable CRR/damage/PPR ODE layer to predict PPR(N), liquefaction risk and censored N_liq under site-held-out validation.
