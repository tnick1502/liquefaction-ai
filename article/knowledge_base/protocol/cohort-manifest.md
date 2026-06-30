---
tags: [protocol, dataset, cohort, manifest]
date: 2026-06-30
---
# Cohort manifest — RAW vs ANALYTIC risk set

Связано: [[evaluation-protocol]] · [[reviewer-remarks-fixes]] · [[metrics]]

Различаем **сырую когорту** (всё, что извлечено из пиклов) и **аналитический risk set** (то, на чём
обучаются/оцениваются модели после landmark-протокола). КОНКРЕТНЫЕ ЧИСЛА здесь НЕ фиксируются —
они зависят от источника `data/sites` и от N₀ (config `prefix_landmark_cycles`) и **генерируются на
каждый прогон** функцией `liquefaction_ai.data.build_cohort_manifest(population, raw_count)` →
`data/dataset/cohort_manifest.json`. Ниже — ОПРЕДЕЛЕНИЯ срезов (что считать), не значения.

| Срез | Что это |
|---|---|
| **RAW specimens** | все извлечённые опыты (`extract_test` по всем объектам/типам) |
| Исключено: событие до landmark N₀ | событие уже произошло к моменту обусловливания (не прогнозируемо из префикса) |
| Исключено: цензура до landmark N₀ | субъект не под наблюдением в N₀ и не имеет continuation-target |
| **ANALYTIC landmark risk set** | event-free и under observation в N₀ — это и есть когорта обучения/оценки |
| — разжижение (liq) | label=1 (первое пересечение ru≥0.95 в окне ≤H) |
| — нет разжижения (non-liq) | право-цензура на фактическом last_obs |
| Объектов / площадок (raw / analytic) | единица group-split = `site_id`; analytic sites — только с landmark-eligible образцами |
| Режимы seismic / storm | по типу воздействия |
| CRR измерена: образцов / объектов | образцы с измеренной кривой CRR(N) и число площадок за ними (held-out фолд может содержать лишь 1 CRR-объект → CRR claim осторожный) |

**Единый горизонт задачи:** входная/query-сетка фиксирована `1…3000` для ВСЕХ опытов (uniform
prediction horizon); `last_obs` определяет только target-маску/censor time. `N_max`/`cycles_count`
**удалены из входных признаков** (в реальных опытах ≈ длительность → утечка).

**Risk-метка по ОКНУ НАБЛЮДЕНИЯ (без absorbing-гипотезы).** `risk_label_observed` = разжижение ИЛИ
non-liq, доведённый до ≥H без пересечения. Non-liq, остановленные до H, имеют неизвестный by-H исход и
исключаются только из risk-BCE/метрик (не из event-time).

**Event-time цензура независима от risk-mask.** Каждый landmark-eligible субъект даёт точный event-time
или валидную нижнюю границу `N_liq > last_obs` (`nliq_censor_valid`). Физические режимы (liquefied /
stabilized / unfinished) — отдельные regime-маски, тоже независимы от risk-mask.

**Проспективный audit** (`audit_horizon_negatives`, causal envelope): среди non-liq, ПЛОСКИХ в раннем
окне 400–500 циклов, считает, сколько позже пересекли порог и сколько позже выросли. Назначение —
эмпирически проверить, что «нет события by H» держится, а «absorbing stable state» — нет (поэтому
negative-метку строим по факту наблюдения до горизонта, а не по «стабилизации»). Числа — в манифесте.
