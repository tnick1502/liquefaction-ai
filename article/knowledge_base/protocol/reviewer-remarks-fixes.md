---
tags: [protocol, reviewer, fixes, code-review]
date: 2026-06-28
---
# Разбор критических замечаний рецензента и фиксы (по логике/коду)

Связано: [[p0-findings]] · [[object-split-policy]] · [[evaluation-protocol]] · [[metrics]]

| # | Замечание | Оценка | Что сделано |
|---|---|---|---|
| 1 | Leaderboard невалиден из-за устаревшего артефакта (config.json без prefix-флагов; 24% утечки) | **Верно** (операционно) | Код `config.py` уже содержит флаги; нужна **перематериализация** данных (1_x) + переобучение. Не кодовый баг — шаг запуска. Зафиксировано в [[p0-findings]] и README. |
| 2 | Primary CV не stratified: при страте <5 откат к KFold; фолды без CRR | **Верно, баг** | Переписан `make_grouped_cv_folds`: round-robin CRR-объектов первыми → **CRR в каждом тест-фолде** при любых стратах. Проверено: CRR/фолд = [1,1,1,1,1]. |
| 3 | Validation слабая: 1 CRR-объект, часто positive-only | **Верно** | `_pick_balanced_val`: val ≥2 объектов **обоих классов** + CRR-объект. Проверено на синтетике. |
| 4 | «conditional flow uncertainty» ≠ инференс (eps=0, один проход) | **Верно, критично** | Добавлен `DPIFlow.predictive(batch, K)` — **MC-пропагация через flow** (epistemic от θ + aleatoric от logvar-головы); `collect_outputs` зовёт его при `mc_samples_eval>1`. EVT-семейство — на гетероскедастичной голове (отмечено честно). |
| 5 | CV оценивает full-horizon Traj_RMSE, не post-prefix | **Верно** | `cross_validation` передаёт в P³ **continuation-метрики** (`Traj_RMSE_continuation_balanced/_worst`); p3_ranking их предпочитает. Headline-таблица уже на continuation. |
| 6 | Primary CV только 4 модели (нет сильных конкурентов) | **Верно** | `DEFAULT_MODELS` расширен: **Transformer, Neural Spline Flow, CatBoost, GRU, TCN** + флагманы + PINN. CatBoost через нативный `.fit` (не torch-цикл). |
| 7 | CI некорректны: 1.96·std/√5, фолды не независимы, псевдорепликация | **Верно** | Добавлен `object_cluster_bootstrap` (кластеры=объекты/площадки) на **pooled OOF**; для CI берётся один OOF-проход (без псевдорепликации repeats). Наивный CI помечен как справочная дисперсия. NB 3_5 использует cluster bootstrap как primary. |
| 8 | Runner неидемпотентен (`mode='a'` → дубли) | **Верно (для удалённого скрипта)** | В ноутбуках CSV пишутся **overwrite** (concat в памяти → один write). Добавлен `cv_grouped_run_meta.json` (prefix-флаг, seed, n_splits, n_repeats, mc). |
| 9 | CRR claim слишком широкий (71 проба, 1 объект) | **Верно** | Сплит теперь кладёт CRR-объект в каждый тест-фолд → CRR оценивается по нескольким объектам в CV. Claim формулировать как **exploratory CRR recovery** (см. [[../tracks/AAAI-27]] limitations). |
| 10 | Physics evidence узкое (PhysViol=0 by construction) | **Частично в коде** | Добавлены абляции **`blackbox_cummax`** и **`blackbox_raw`**: показывают, что выигрыш не сводится к cummax. **TODO (документировано):** onset-coherence CRR(N_liq)≈CSR, parameter plausibility, sensitivity — требуют доп. инфраструктуры (диагностические метрики поверх outputs `crr`/`theta`). |

## Порядок прогона (после фиксов)
1. Перематериализовать `data/*` (1_x, strict prefix) → 2. Переобучить (2_x) → удалить старые headline CSV →
3. `3_4` (repeated balanced grouped CV + MC-uncertainty) → `3_5` (Wilcoxon+Holm + **object-cluster bootstrap**) →
4. `3_6` (абляции, включая blackbox_cummax/raw) → `3_7` (фигуры) → `3_8` (consistency+P³-sensitivity).

## Осталось (для 7/10 → выше)
- Onset-coherence `CRR(N_liq)≈CSR` и parameter-plausibility как метрики (P1-доп).
- LOO-20 как secondary failure-analysis (флаг `RUN_LOO` в 3_4).
- Claims/abstract — только ПОСЛЕ прогона на перематериализованных данных.

---

## Раунд 2 — фиксы по результатам первого прогона
| # | Замечание | Фикс |
|---|---|---|
| 1 | n_repeats=1, не repeated | NB 3_4 default **N_REPEATS=3** (код повторов был готов) |
| 2 | калибровка не в CV → Coverage@90≈0.76 | `evaluate_fold` вызывает **fit_interval_scale на VAL каждого фолда** (torch-модели) |
| 3 | headline на наивном fold-CI | `headline_table(summary, cluster_df)` берёт **CI из object-cluster bootstrap**; NB 3_7 передаёт его |
| 4 | Wilcoxon псевдорепликация (n≈800) | `paired_significance(cluster='object')` по умолчанию: агрегат по объекту → тест по ~N объектов |
| 5 | нет no-prefix/no-aux stress | абляции **no_prefix / no_aux** через `stress_split` |
| 6 | абляции на 1 фолде | NB 3_6 default **FOLDS=[0,1,2]** |
| 7 | 2 теста красные | allowlist: добавлены `publication_headline_*.csv`; **PVR-тест оставлен строгим** (monotone_clip→0 by construction; падение = устаревший `full_leaderboard.csv`, позеленеет после перепрогона 3_1) |
| 8 | CRR не в primary CV | METRIC_KEYS += `CRR_RMSE, N_CRR_test, N_CRR_objects` |
| 9 | герой неочевиден | claim → Pareto-balance (см. [[../formulations/positioning-and-title]]) |
| 10 | MC: нет sens по K; CRR от последнего сэмпла | `predictive` теперь **усредняет CRR** по MC; K-sensitivity (4/8/16/32) — через `config.mc_samples_eval` (прогон-параметр) |

**Per-sample coverage90** добавлен в `compute_metrics` → cluster-bootstrap покрывает калибровку.
