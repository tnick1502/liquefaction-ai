---
tags: [protocol, P0, findings, executed]
status: in-progress
date: 2026-06-28
---
# P0 — журнал решений (методы)

Связано: [[evaluation-protocol]] · [[object-split-policy]] · [[cohort-manifest]] · [[../formulations/significance-and-data-value]] · `../../recommendations.md` §2

> Числовые находки/результаты сюда не заносятся (см. политику в [[../00_INDEX]]); ниже — методические решения.

## ✅ P0-c — Prefix leakage: НАЙДЕНА и ИСПРАВЛЕНА
**Суть проблемы:** при наивном префиксе часть разжижающихся опытов «показывала» модели само событие
(вход доходил до/за onset, ru≥порога во входе) → артефактно завышенные risk-метрики.

**Текущий фикс (метод):** outcome-independent префикс по **landmark-циклу N₀** (`prefix_mode="landmark"`,
`prefix_landmark_cycles`) — наблюдения только до N₀, гарантированно до возможного onset (исторически
была версия `strict_preonset` с обрезкой до onset; сейчас канон — landmark, см. [[evaluation-protocol]]).
Событие N_liq определяется на поцикловой огибающей независимо от сетки.

⚠️ Фикс зашит в материализацию (`seq_inputs`/`prefix_summary`) → требует регенерации `data/dataset`.

**Ещё к P0-c:** добавить стресс-тесты **no-prefix** и **prefix-length sweep** (есть в [[ablations]] п.6).

## ✅ Сплит — РЕАЛИЗОВАН
Primary = **stratified grouped K-fold по площадкам (`site_id`)**, secondary = **LOO-site**. Критический
разбор и почему «мелкие→test» отвергнуто — [[object-split-policy]].
- `data/splits.py`: `make_grouped_cv_folds(...)` (балансировка CRR×liq, val с CRR-объектом и обоими
  классами) и `make_loo_object_folds(...)`; хук `prepare_benchmark_dataset(..., precomputed_split=fold)`.
- **Контракт (тесты):** каждая площадка тестируется ровно раз; train/val/test не пересекаются по
  `site_id`; CRR-объект в val каждого фолда; без утечки.

## ✅ P0-d — publication-эпохи (не demo)
Кросс-валидация (ноутбук **3_4** / `evaluation.cross_validation`) использует `publication_*` эпохи + cosine + early stopping; `QUICK=True` — только дымовой тест.

## ⏳ P0-a — прогон CV с CI: ЛОКАЛЬНО (песочница не тянет torch)
Среда: `.venv` сломан, torch не ставится в 45с. Команды для автора:
```
# primary: stratified grouped 5-fold
Ноутбук 3_4: build_folds → evaluate_fold по всем фолдам (N_REPEATS=3)
                 → cv_grouped_{raw,samples,summary}.csv

# secondary: LOO по 20 объектам
Ноутбук 3_4: RUN_LOO=True (secondary)

```
Выдаёт mean ± 95% CI по фолдам для P3_Core, N_liq_logMAE, Traj_RMSE, AUROC, **AUPRC**, Brier, PhysViol.

## ✅ P0-b — значимость/bootstrap: РЕАЛИЗОВАНО (валидировано на синтетике)
- CV (ноутбук **3_4**) пишет **per-sample** выходы → `results/analysis_tables/cv_{grouped,loo}_samples.csv` (строка на (fold,model,образец); пара (fold,sidx) одинакова для всех моделей → корректное парное сравнение).
- Значимость (ноутбук **3_5** / `evaluation.significance`, только numpy/pandas):
  1. **Парный Wilcoxon signed-rank** (нормальная аппрокс. с поправкой на непрерывность, обработка нулей/связей) ref=DPI-Flow vs каждая модель по `traj_rmse_continuation` и `nliq_log_err`; **Holm–Bonferroni** по семейству; effect size = **rank-biserial** + медиана разницы с 95% bootstrap-CI.
  2. **Stratified bootstrap-CI** (ресэмпл по фолдам, 1000) для **AUROC/AUPRC/Brier/ECE** по модели и для разницы с опорной (флаг значимости = CI разницы не содержит 0).
- Значимость считается на pooled OOF (object-cluster), числовые результаты собираются после прогона (не в KB).

Запуск: ноутбук **3_5** после 3_4.
```
```
→ `significance_{grouped,loo}_pairwise.csv` и `_bootstrap.csv`
