---
tags: [protocol, P0, findings, executed]
status: in-progress
date: 2026-06-28
---
# P0 — журнал выполнения (фактические результаты)

Связано: [[evaluation-protocol]] · [[object-split-policy]] · [[../formulations/significance-and-data-value]] · `../../recommendations.md` §2

## ✅ P0-c — Prefix leakage: НАЙДЕНА и ИСПРАВЛENA
**Находка** (numpy по `data/dataset/arrays.npz`): у **152/637 = 23.9%** разжижающихся префикс доходил до/за onset; в 23.8% вход уже достигал ru≥0.95 — т.е. модель «видела» само разжижение. Это и есть причина AUROC≈1.0.

**Фикс (реализован в коде):**
- `config.py`: флаги `prefix_strict_preonset=True`, `prefix_onset_threshold=LIQ_THRESHOLD`, `prefix_onset_margin=1`, `prefix_min_len=3`.
- `data/real_adapter.py`: новый `strict_pre_onset_prefix_mask(...)` + `build_observed_prefix(...)` обрезает префикс строго до onset (гарантия: все индексы префикса < onset_idx ⇒ ru < порога).
- `data/synthetic.py`: тот же helper по истинному onset (`r_true≥LIQ_THRESHOLD`).

**Верификация (реальные данные, тот же helper):** LEAK-A **152→0**, LEAK-B(ru≥0.95) **→0**; длины префикса 3–12 (медиана 12, mean 10.3); сверхбыстрых (<3) нет; укорочено 153 разжижающихся опыта.

⚠️ **Чтобы фикс попал в обучение — нужно ПЕРЕматериализовать артефакт локально** (префикс зашит также в `seq_inputs`/`prefix_summary`): перегенерировать `data/*` через пайплайн (1_x ноутбуки / `run_all.py`) с обновлённым кодом. Затем **ожидаемо** AUROC/AUPRC честно просядут.

**Ещё к P0-c:** добавить стресс-тесты **no-prefix** и **prefix-length sweep** (есть в [[ablations]] п.6).

## ✅ Сплит — РЕАЛИЗОВАН (решение автора)
Primary = **stratified grouped 5-fold по объектам**, secondary = **LOO-object**. См. критический разбор и почему «мелкие→test» отвергнуто: [[object-split-policy]].
- `data/splits.py`: `make_grouped_cv_folds(...)` (StratifiedKFold по объектному страту = CRR×liq, fallback KFold; val с CRR-объектом) и `make_loo_object_folds(...)`; + хук `prepare_benchmark_dataset(..., precomputed_split=fold)`.
- **Юнит-проверка (синт. meta, 20 объектов):** каждый объект тестируется ровно раз, train/val/test по объектам не пересекаются, test-фолды непересекающиеся, CRR-объект в val каждого фолда; LOO — 20 фолдов по 1 объекту, без утечки. ✓

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
- **Дымовой тест на синтетике пройден:** DPI-Flow значимо лучше EVT/PINN по траектории и N_liq (p_holm=0, rank-biserial 0.90–1.0, CI разницы > 0); bootstrap отделяет PINN по AUROC/AUPRC/Brier.

Запуск: ноутбук **3_5** после 3_4.
```
```
→ `significance_{grouped,loo}_pairwise.csv` и `_bootstrap.csv`
