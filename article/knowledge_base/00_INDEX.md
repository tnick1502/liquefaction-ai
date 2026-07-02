---
tags: [moc, index]
---
# 📚 База знаний: публикации DPI-Flow / разжижение

Map of Content (MOC) для подготовки публикаций. Основано на `../publication_plan.md`, критически переработанной литературе (`../recomendations/*.docx`) и сводных `../recommendations.md`.

> **Политика базы знаний.** KB описывает **только методы, протокол, определения и литературу**.
> **Числовые результаты экспериментов сюда НЕ заносятся** — они собираются отдельно ПОСЛЕ чистого
> прогона на зафиксированной когорте (regenerate → retrain → eval) и хранятся в `results/`. Любое
> конкретное значение метрики/исхода в KB — устаревший артефакт, подлежит удалению.

## 🎯 Треки
- [[tracks/AAAI-27]] — **главный**, Main Technical Track (дедлайн abstract 21.07.2026)
- [[tracks/NeurIPS-workshop-topology]] — topology / early-warning (параллельно, не смешивать)
- [[tracks/Q1-journal]] — полный framework (позже)

## 🧭 Два направления подачи (упор по venue)
- [[literature/_reading-aaai]] — **упор на AAAI**: ML-метод вперёд, геотехника компактно
- [[literature/_reading-geotech]] — **упор на геотехнику/журнал**: инженерная физика + state-of-practice + российская школа вперёд

## ✍️ Формулировки (готовый текст)
- [[formulations/significance-and-data-value]] — **значимость задачи + ценность архива** (дорогой эксперимент → ценные данные; ответ на «мало данных»)
- [[formulations/positioning-and-title]] — позиционирование, title, keywords, главный claim
- [[formulations/thesis-and-contributions]] — тезис + 5 вкладов
- [[formulations/abstract-draft]] — черновик abstract v0
- [[formulations/intro-paragraphs]] — 4 абзаца введения
- [[formulations/methods-paragraphs]] — 8 абзацев методов

## 🧠 Методы: кастомные нейросети
- [[methods/custom-networks]] — **подробное описание архитектур** DPI-Flow / DPI-EVT / EVT-NeuralSSM: уравнения, лоссы, физические инварианты, curve-first N_liq, общая дифференцируемая машинерия (без результатов)

## 🔬 Протокол и оценка
- [[protocol/evaluation-protocol]] — **P0**: grouped CV, multiseed, значимость, anti-leakage
- [[protocol/metrics]] — **P2**: AUPRC, calibration, lead-time, CRPS/NLL, per-state
- [[protocol/ablations]] — **P1**: 7 абляций к вкладам
- [[protocol/baselines]] — что заявить, чтобы не снесли
- [[protocol/claim-to-citation]] — карта тезис → ссылка
- [[protocol/object-split-policy]] — политика разбиения по объектам (критика «мелкие→test»)
- [[protocol/p0-findings]] — журнал P0 (утечка префикса, сплиты, значимость)
- [[protocol/reviewer-remarks-fixes]] — разбор замечаний рецензента и фиксы (раунды 1–2)
- [[protocol/final-run-settings]] — настройки финального прогона (серьёзное обучение + грид-сёрч)
- [[protocol/notebooks-consolidation]] — карта «скрипты → ноутбуки/библиотека» (аудит)

## 📖 Литература (atomic-заметки с DOI)
**State-of-practice geotech:** [[literature/seed-idriss-1971]] · [[literature/youd-2001]] · [[literature/boulanger-idriss-2016]] · [[literature/cetin-2004]] · [[literature/moss-2006]] · [[literature/kayen-2013]]
**Мотивация / критика AI:** [[literature/maurer-sanger-2023]] · [[literature/jas-dodagoudar-2023-review]] · [[literature/nasem-2021]] · [[literature/brandenberg-2020]]
**Физика PPR/CRR:** [[literature/seed-martin-lysmer-1976]] · [[literature/cetin-bilge-2012]] · [[literature/cetin-2018]]
**Censored event-time:** [[literature/cox-1972]] · [[literature/nafday-2010]] · [[literature/katzman-2018-deepsurv]]
**Physics-informed / ODE / flows:** [[literature/karniadakis-2021]] · [[literature/raissi-2019-pinn]] · [[literature/chen-2018-neuralode]] · [[literature/rezende-mohamed-2015]] · [[literature/kingma-welling-2014]]
**AAAI-якоря / calibration:** [[literature/wang-2024-aaai]] · [[literature/tomani-buettner-2021-aaai]] · [[literature/tao-2025-aaai]] · [[literature/guo-2017-calibration]] · [[literature/romano-2019-cqr]]
**Метрики:** [[literature/brier-1950]] · [[literature/fawcett-2006]] · [[literature/davis-goadrich-2006]] · [[literature/roberts-2017]]
**Tabular baselines:** [[literature/prokhorenkova-2018-catboost]] · [[literature/gorishniy-2021-fttransformer]] · [[literature/arik-pfister-2021-tabnet]]
**ML-конкуренты:** [[literature/sanger-2025-mechanics]] (ближайший) · [[literature/ml-competitors-misc]]

### ➕ Добавлено (свежее, DOI проверены через Crossref)
**AAAI-сторона:** [[literature/huang-2025-cuqds-aaai]] (AAAI-25, conformal+shift) · [[literature/anumasa-2022-latent-time-node-aaai]] (AAAI-22, latent-ODE) · [[literature/klotergens-2024-functional-latent-dynamics]] (2024) · [[literature/angelopoulos-2023-conformal-pid]] · [[literature/auer-2023-hopcpt]] (NeurIPS-23 conformal-TS) · [[literature/nagpal-2021-deep-survival-machines]] (deep survival)
**Геотех-сторона:** [[literature/guo-2025-dl-transfer-liquefaction]] (2025, transfer) · [[literature/demir-2024-ensemble-liquefaction]] (2024, ensembles)
**🌍 Global high-impact (новое):** [[literature/dinh-2017-realnvp]] · [[literature/durkan-2019-neural-spline-flows]] · [[literature/zhou-2021-informer-aaai]] · [[literature/lim-2021-temporal-fusion-transformer]] · [[literature/lu-2021-deeponet]] · [[literature/abdar-2021-uq-review]] · [[literature/phoon-zhang-2022-future-ml-geotech]] · [[literature/baghbani-2022-ai-geotech-review]]
**🇷🇺 Российская школа (МГСУ):** [[literature/ter-martirosyan-2025-groundwater-dynamic-stability]] · [[literature/ter-martirosyan-2023-depth-liquefaction-potential]] · [[literature/sidorov-2024-soil-models-liquefaction]]

## ⚙️ Как пользоваться в Obsidian
Открыть `knowledge_base/` как vault. Заметки связаны wikilinks — Graph View покажет кластеры: треки ↔ протокол ↔ литература. Atomic-заметки литературы дублируют DOI в frontmatter (`doi:`) для экспорта в .bib.

## 🚦 Приоритеты (из ../recommendations.md)
**P0** статистика+leakage → **P1** абляции → **P2** метрики/фигуры → текст → repro checklist.
