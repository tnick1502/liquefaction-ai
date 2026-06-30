---
tags: [formulation, significance, data-value, motivation, key]
---
# Значимость задачи и ценность архива (рамка «дорогой эксперимент → ценные данные»)

Связано: [[positioning-and-title]] · [[thesis-and-contributions]] · [[intro-paragraphs]] · [[../protocol/evaluation-protocol]] · [[../literature/phoon-zhang-2022-future-ml-geotech]] · [[../literature/maurer-sanger-2023]] · [[../literature/nasem-2021]]

## Суть (one-paragraph)
Разжижение — высокорисковый отказ грунта: масштабный материальный ущерб, угроза инфраструктуре и человеческим жизням ([[../literature/nasem-2021]]). При этом **лабораторный циклический опыт до разжижения — дорогой и долгий**: подготовка образца, насыщение, консолидация и само циклическое нагружение растягиваются на дни–недели и требуют специализированного оборудования. Поэтому данные по разжижению **внутренне дефицитны** — это свойство предметной области, а не небрежность сбора. На этом фоне **архив реальных объектов в этой работе (1093 опыта, 20 объектов, измеренный CRR на подмножестве) — высокоценный и в рамках поставленной задачи исчерпывающий ресурс**: он покрывает доступные объекты/ИГЭ целиком, состоит из настоящих, а не синтетических кривых, и фиксирует именно ту наблюдаемую динамику PPR/N_liq, ради которой ставится прогноз.

## Почему это переворачивает критику «мало данных»
Стандартное возражение рецензента — «N мал». Грамотный ответ: **каждый образец здесь — дорогая ground-truth-точка**, полученная неделями физического эксперимента, а не дешёвая строка из веб-скрейпа. Тогда научный вклад смещается с «обучили на больших данных» на **«извлекли максимум сигнала из каждого дорогого опыта»**. Это ровно тот режим *data-centric, small-and-precious-data geotechnics*, который описывают [[../literature/phoon-zhang-2022-future-ml-geotech]]: малые, «дорогие», site-specific выборки требуют сильных индуктивных смещений и честной оценки, а не больших моделей.

## Методологические следствия (каждое — прямая опора вклада)
1. **Физически структурированный prior** (ODE-слой + параметрическая идентификация θ) — оправдан именно дефицитом данных: сильное inductive bias заменяет отсутствующие миллионы примеров. → [[methods-paragraphs]]
2. **Censoring-aware обучение** — landmark-eligible liquefied дают точный event-time, а все non-liq дают правую цензуру на фактическом last_obs; stabilized/unfinished дополнительно разделяют trajectory-robustness. → [[../protocol/ablations]] · [[../literature/cox-1972]]
3. **Prefix-conditioning имеет прямую экономическую отдачу:** прогноз исхода по раннему фрагменту = потенциальная возможность **сокращать будущие многонедельные опыты** (ранняя остановка/триаж). Это не только ML-постановка, но и практический payoff для дорогого эксперимента. → [[thesis-and-contributions]]
4. **Object-held-out + multi-seed CI** — статистическая строгость нужна *именно потому*, что выборка мала и «драгоценна»: завышать качество leakage'ем недопустимо, дисперсию надо показывать честно. → [[../protocol/evaluation-protocol]] · [[../literature/roberts-2017]]
5. **Калиброванная неопределённость** — решения по дорогим/редким данным принимаются под риском; нужен не точечный прогноз, а калиброванный интервал/риск. → [[../protocol/metrics]]

## Честная граница (без overclaim)
«Исчерпывающий» — **в рамках задачи**: полный охват *доступных объектов и ИГЭ данной кампании*, а не всех грунтов/режимов мира. Это один лабораторный источник; перенос lab→field не заявляется; обобщение за пределы представленных типов грунта/нагружения ограничено. Формулировать как **comprehensive within scope**, а не **universal** — иначе рецензент справедливо укажет на узость. Эта оговорка усиливает доверие, а не ослабляет вклад.

## Готовые формулировки (EN, для intro/abstract/significance)
- *Soil liquefaction is a high-consequence failure mode — driving large economic losses and threatening lives and infrastructure — yet each cyclic laboratory test that characterizes it can take days to weeks and specialized equipment to run. Liquefaction data are therefore intrinsically scarce.*
- *Against this cost, our archive of real cyclic tests (1093 specimens across 20 sites, with measured CRR on a subset) is a high-value, scope-exhaustive resource: it covers the available objects in full, consists of real rather than synthetic trajectories, and records exactly the PPR/N_liq dynamics we set out to forecast.*
- *This reframes the small-sample setting: every datum is expensive ground truth, so the contribution is to extract maximal signal per experiment via physics structure, censoring-aware training, and prefix-conditioned forecasting — and, in turn, to forecast outcomes early enough to potentially shorten future weeks-long tests.*
- *We make a bounded claim: the dataset is comprehensive within its scope (the available objects and soil units of this campaign), not universal; lab-to-field transfer is not claimed.*

## Куда вставлять в статье
- **Introduction §1** (stakes) и **§2→§3** (scarcity → почему physics-structured + censoring).
- **Problem Setup / Experimental Setup** — абзац о ценности и охвате архива.
- **Limitations** — «comprehensive within scope, not universal».
- **Broader impact / significance** (AAAI ценит) — ущерб/жизни + practical payoff ранней остановки опытов.
