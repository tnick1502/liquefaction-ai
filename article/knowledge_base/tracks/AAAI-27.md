---
tags: [track, aaai, primary]
status: active
---
# Трек 1 — AAAI-27 (Main Technical Track)

**Фокус:** DPI-Flow как компактный ML-вклад. См. полные рекомендации: `../../recommendations.md`.

## Дедлайны
- Abstract: **21 июля 2026** · Full paper: **28 июля 2026** · Supplementary/code: **31 июля 2026**.
- Формат: **7 страниц** + неограниченные references.

## Идея
Вероятностная physics-constrained модель: по раннему фрагменту испытания + свойствам грунта прогнозирует onset, N_liq, траекторию PPR и неопределённость.

## Структура (7 стр.)
Intro 0.75 · Problem Setup 0.75 · Method 1.5 · Experimental Setup 1.0 · Results 1.5 · Ablations & Robustness 1.0 · Limitations & Conclusion 0.5.

## Ключевые узлы
- Значимость + ценность данных: [[../formulations/significance-and-data-value]]
- Позиционирование/title: [[../formulations/positioning-and-title]]
- Тезис/вклады: [[../formulations/thesis-and-contributions]]
- Intro/Methods абзацы: [[../formulations/intro-paragraphs]] · [[../formulations/methods-paragraphs]] · [[../formulations/abstract-draft]]
- Протокол (P0): [[../protocol/evaluation-protocol]]
- Метрики (P2): [[../protocol/metrics]]
- Абляции (P1): [[../protocol/ablations]]
- Baselines: [[../protocol/baselines]]
- Citations: [[../protocol/claim-to-citation]]

## Limitations (писать заранее)
- monotonicity assumed; CRR measured for limited objects (**N_CRR_objects=1 в held-out test → claim осторожный**); prefix-conditioned setup упрощает задачу (закрыто стрессами); lab-to-field transfer not claimed; topology — supplement/future work.

## Риски и закрытие
| Риск | Закрытие |
|---|---|
| Выглядит как application paper | title/abstract/contributions — про ODE/flow/event-time/calibration; геотехника → benchmark |
| AUROC≈1 → leakage/prefix-shortcut | no-prefix stress, object-held-out, проверка «префикс до onset» ([[../protocol/evaluation-protocol]]) |
| «Мало данных / N мал» | переформулировать через [[../formulations/significance-and-data-value]]: дорогой эксперимент → scope-exhaustive архив; вклад = max сигнала на опыт |
| Слабые baselines | GBDT + deep tabular + sequence + survival + classical triggering + Sanger 2025 ([[../protocol/baselines]]) |
| Physics layer = hand-crafted | подать как generalizable abstraction: constrained param inference + diff. simulator + event-time loss + ablation |
| Reproducibility | заполнить `AuthorKit27/ReproducibilityChecklist.tex` |
