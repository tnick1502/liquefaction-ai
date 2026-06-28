---
tags: [reading-list, direction/aaai, moc]
---
# 🤖 Reading-list — упор на AAAI (ML/AI-вклад)

Для подачи в AAAI **геотехнику держим компактно** (мотивация), а на первый план — методную линию: physics-structured probabilistic forecasting, neural ODE, normalizing flows, conformal UQ, censored event-time. Порядок = приоритет цитирования в AAAI-версии.

## Ядро метода (must-cite)
1. [[chen-2018-neuralode]] — differentiable ODE
2. [[rezende-mohamed-2015]] — normalizing flows над θ
3. [[karniadakis-2021]] · [[raissi-2019-pinn]] — physics-informed framing (контраст: hard structure vs soft penalty)
4. [[kingma-welling-2014]] — latent-variable inference

## Свежие AAAI-нативные / SOTA-якоря ⭐ (новое)
5. [[huang-2025-cuqds-aaai]] — **AAAI-25**: conformal UQ под distribution shift, trajectory → наш центральный якорь
6. [[anumasa-2022-latent-time-node-aaai]] — **AAAI-22**: latent-time neural ODE
7. [[wang-2024-aaai]] — **AAAI-24**: physics-informed representation + risk
8. [[klotergens-2024-functional-latent-dynamics]] — 2024: latent dynamics для нерегулярных рядов

## Uncertainty / calibration / conformal
9. [[guo-2017-calibration]] · [[romano-2019-cqr]] — классика ECE + conformal
10. [[angelopoulos-2023-conformal-pid]] · [[auer-2023-hopcpt]] — **свежее conformal-TS семейство (NeurIPS-23)**
11. [[tomani-buettner-2021-aaai]] · [[tao-2025-aaai]] — AAAI calibration line
12. [[brier-1950]] · [[fawcett-2006]] · [[davis-goadrich-2006]] — метрики

## Censored event-time (N_liq)
13. [[cox-1972]] — база
14. [[katzman-2018-deepsurv]] · [[nagpal-2021-deep-survival-machines]] — **deep survival baselines** (новое: DSM)

## Tabular / sequence baselines
15. [[prokhorenkova-2018-catboost]] · [[gorishniy-2021-fttransformer]] · [[arik-pfister-2021-tabnet]]

## Протокол (AAAI ждёт строго)
16. [[roberts-2017]] — grouped/site-held-out → [[../protocol/evaluation-protocol]]

## 🌍 Global high-impact (новое — DOI проверены)
- **Flows (наши бейзлайны):** [[dinh-2017-realnvp]] (Real NVP) · [[durkan-2019-neural-spline-flows]] (NSF — сюжет admissibility vs flexibility)
- **TS forecasting:** [[zhou-2021-informer-aaai]] (AAAI-21 Best Paper) · [[lim-2021-temporal-fusion-transformer]] (quantile multi-horizon)
- **Scientific ML / operators:** [[lu-2021-deeponet]]
- **UQ обзор:** [[abdar-2021-uq-review]]

## Геотехника — минимум для мотивации (1 абзац)
[[maurer-sanger-2023]] (gap) · [[seed-idriss-1971]] · [[boulanger-idriss-2016]] · [[sanger-2025-mechanics]] (ближайший конкурент)

> Полные рекомендации: `../../recommendations.md` · трек: [[../tracks/AAAI-27]]

> 🎯 Рамка значимости/ценности данных (применять в Intro+Significance+Limitations): [[../formulations/significance-and-data-value]]
