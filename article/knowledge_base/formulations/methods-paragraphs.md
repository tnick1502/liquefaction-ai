---
tags: [formulation, methods, draft]
---
# Готовые абзацы методов

Связано: [[../protocol/ablations]] · [[../protocol/claim-to-citation]]

1. **Физическая часть.** PPR как накопительная циклическая динамика, связанная с CSR/CRR и повреждением/деградацией сопротивления. → [[../literature/seed-martin-lysmer-1976]] [[../literature/cetin-bilge-2012]]
2. **Латентные θ.** Интерпретировать как **физически ограниченные** параметры (softmax/sigmoid-проекции в допустимые диапазоны), не произвольный embedding. Связь θ → CRR(N), damage z(N), trigger g(N), PPR(N).
3. **Дифференцируемый ODE-layer.** Динамика не декодируется black-box последовательностью, а проходит через структурированную вычислительную модель. → [[../literature/chen-2018-neuralode]] [[../literature/karniadakis-2021]] (контраст с [[../literature/raissi-2019-pinn]]: hard structure vs soft penalty)
4. **Conditional affine flow.** Повышает гибкость posterior над θ, сохраняя end-to-end обучение. → [[../literature/rezende-mohamed-2015]] [[../literature/kingma-welling-2014]]
5. **Censored N_liq.** Если onset не наступил в горизонте опыта — **право-цензурированное** наблюдение, не обычная регрессионная метка. → [[../literature/cox-1972]] [[../literature/nafday-2010]]
6. **Loss:** `L = L_traj + L_risk + L_censored_Nliq + L_aux + L_KL/flow + L_physics`. Conformal/post-hoc interval calibration. → [[../literature/romano-2019-cqr]] [[../literature/guo-2017-calibration]]
7. **Vs из small-strain G0** (не из secant-модуля петли) — деталь валидности фич; см. memory `liquefaction-digitrock-sources`. Перекликается с [[../literature/kayen-2013]].
8. **Валидация:** site-held-out/grouped split (ни один объект в train и test). → [[../literature/roberts-2017]]
