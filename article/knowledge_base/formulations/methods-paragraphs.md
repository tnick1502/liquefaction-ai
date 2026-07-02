---
tags: [formulation, methods, draft]
---
# Готовые абзацы методов

Связано: [[../protocol/ablations]] · [[../protocol/claim-to-citation]]

1. **Физическая часть.** PPR как накопительная циклическая динамика, связанная с CSR/CRR и повреждением/деградацией сопротивления. → [[../literature/seed-martin-lysmer-1976]] [[../literature/cetin-bilge-2012]]
2. **Латентные θ.** Интерпретировать как **физически ограниченные** параметры (softmax/sigmoid-проекции в допустимые диапазоны), не произвольный embedding. Связь θ → CRR(N), damage z(N), trigger g(N), PPR(N).
3. **Дифференцируемый ODE-layer.** Динамика не декодируется black-box последовательностью, а проходит через структурированную вычислительную модель. → [[../literature/chen-2018-neuralode]] [[../literature/karniadakis-2021]] (контраст с [[../literature/raissi-2019-pinn]]: hard structure vs soft penalty)
4. **Conditional coupling flow (RealNVP).** Latent-зависимые coupling-слои с корректной density-objective (log-det Jacobian) — настоящий conditional RealNVP, а не affine-смещение контекста. Разброс θ — это **conditional (амортизированный) posterior по входу**, а НЕ байесовская эпистемическая неопределённость по параметрам модели; формулировать соответственно. Headline использует нормированную плотность потока (`calibration_steps=0`); test-time градиентная доводка θ — только абляция. → [[../literature/dinh-2017-realnvp]] [[../literature/rezende-mohamed-2015]] [[../literature/kingma-welling-2014]]
5. **Censored N_liq (curve-first).** Если onset не наступил в горизонте опыта — **право-цензурированное** наблюдение, не обычная регрессионная метка. Публикуемый N_liq = линейно-интерполированный момент пересечения порога **собственной средней PPR-кривой** модели (лежит на кривой по построению, coherence-gap=0); выделенная head — auxiliary-предиктор, обучаемый censored-лоссом + consistency к кривой. Единообразно во всех трёх моделях. → [[../literature/cox-1972]] [[../literature/nafday-2010]] · подробно: [[../methods/custom-networks]]
6. **Loss:** `L = L_traj + L_risk + L_censored_Nliq + L_aux + L_KL/flow + L_physics`. Калибровка интервалов: post-hoc **variance-scaling** σ (НЕ conformal); headline coverage = **empirical site-held-out conformal** с object-bootstrap CI (не finite-sample гарантия). → [[../literature/romano-2019-cqr]] [[../literature/guo-2017-calibration]]
7. **Vs из small-strain G0** (не из secant-модуля петли) — деталь валидности фич; см. memory `liquefaction-digitrock-sources`. Перекликается с [[../literature/kayen-2013]].
8. **Валидация:** site-held-out/grouped split (ни один объект в train и test). → [[../literature/roberts-2017]]
9. **Причинный onset/prefix протокол.** Событие/N_liq — первый цикл УСТОЙЧИВОГО пересечения ru≥0.95 на нескольких подряд идущих целых циклах (на сырых пиках). Вход — причинный префикс до landmark N₀ без клиппинга по метке; образцы с наблюдаемым пересечением к N₀ исключаются из risk set (не переписываются), terminal-ambiguous аудируются/исключаются. Это устраняет утечку метки через вход при малом числе площадок.
