---
tags: [protocol, metrics, P2]
---
# Метрики (фокус, не количество)

Связано: [[evaluation-protocol]] · [[../literature/guo-2017-calibration]] · [[../literature/davis-goadrich-2006]] · [[../literature/romano-2019-cqr]]

## Onset (классификация)
- **AUPRC — primary** на известных by-горизонту исходах (маска `risk_label_observed`), AUROC — вторично.
- **Brier** ([[../literature/brier-1950]]).
- **Калибровка: ECE + reliability diagram (фигура)** ([[../literature/guo-2017-calibration]]). **Обязательно на grouped split** — OOD-калибровка обычно ломается; если держится → headline.
- **Lead-time / timeliness:** за сколько циклов до фактического onset поднимается риск → early-warning метрика. Оговорка: `Onset_EarlyWarning_Rate` геймабельна (занижением порога) → сопровождать ложной тревогой, не headline.

## N_liq (event-time, цензура)
- Headline считать **только на нецензурированных** (или censored-aware), явно оговорив. MAE по right-censored атакуем. Обоснование цензуры: [[../literature/nafday-2010]], [[../literature/cox-1972]].
- Метрика: **N_liq_logMAE** (one-sided для цензурированных).

## Trajectory (вероятностная)
- Вперёд **CRPS + NLL** (proper scoring), RMSE — diagnostic. Все траекторные метрики — на СТРОГО post-prefix (continuation) участке.
- **PICP / MPIW** (coverage + interval width) вместе. Conformal: [[../literature/romano-2019-cqr]].
- **Per-state (3-regime)** primary; `Traj_RMSE_worst` — robustness-история; pooled — вторично.

## Site-macro агрегаты (обменочная единица)
- **`Traj_RMSE_continuation_siteMacro`, `N_liq_logMAE_siteMacro`** — сначала среднее ВНУТРИ площадки (`site_id`), потом по площадкам: каждая площадка весит одинаково, не доминируется крупными. `N_sites_test` раскрывает мощность.

## Калибровка (терминология)
- **Headline coverage = empirical site-held-out conformal** (`aggregate_object_conformal`): nonconformity + VAL-калиброванный квантиль на disjoint фолде, эмпирическое покрытие + object-bootstrap CI. Это НЕ finite-sample гарантия.
- **`fit_interval_scale` — variance-scaling** (масштаб σ), НЕ conformal. Множитель-запас = 1.0 (без test-informed подгонки).

## CRR (уникальное)
- CRR_RMSE — **secondary**. `N_CRR_objects` считается по `site_id` (площадкам); при малом числе площадок за измеренной CRR claim осторожный (см. [[../tracks/AAAI-27]] limitations).


## ✅ Статус P2: РЕАЛИЗОВАНО
**Новые метрики в `evaluation/metrics.py` (`compute_metrics`):**
- `N_liq_logMAE_liq` / `N_liq_MAE_liq` — N_liq только на разжижающихся (точные таргеты), прозрачный headline без цензур-эффектов.
- `Onset_EarlyWarning_Rate` — доля разжижающихся, где модель ставит онсет не позже фактического (timely warning).
- `Onset_Timing_Bias_cyc` / `Onset_Timing_MAE_cyc` — сдвиг/ошибка тайминга онсета в циклах (lead-time рамка); + per-sample `onset_timing_bias_cyc`.

**Ноутбук 3_7 (`evaluation.publication`, numpy/pandas/matplotlib, поверх CV-артефактов):**
- **Reliability diagram** калибровки онсет-риска из per-sample (`cv_*_samples.csv`), подпись с ECE → `results/analysis_figs/p2_reliability_*.png/pdf`.
- **Publication headline table** — переупорядочивает метрики под P2-фокус (AUPRC и калибровка вперёд, CRPS, per-state worst, censored + liquefied-only N_liq, physics violations; **AUROC — справочно**), тянет mean±95%CI из `cv_*_summary.csv` → `results/tables/publication_headline_*.csv`.

Запуск: ноутбук **3_7** (после 3_4/3_5). Реализует фокус из раздела выше: AUPRC>AUROC, ECE+reliability, lead-time, CRPS вперёд, per-state primary, liquefied-only N_liq.

## 📊 Публикационные фигуры (English, `evaluation.publication`)
- `reliability_diagram` — калибровка онсет-риска (ECE в подписи).
- `forest_plot` — сравнение моделей по метрике с **object-cluster bootstrap 95% CI** (dot-and-whisker).
- `pareto_plot` — onset (AUPRC) ↔ trajectory (post-prefix RMSE) trade-off; physics-admissible заполнены, structured-модели акцентированы → визуализирует claim о Pareto-балансе.
- `ablation_bars` — вклад компонентов (метрика по вариантам абляции, 'full' выделен).
Встроены в ноутбуки **3_7** (forest/pareto/reliability/headline) и **3_6** (ablation-bars). Всё на английском.
