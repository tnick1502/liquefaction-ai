---
tags: [protocol, validation, P0]
status: TODO (multiseed не прогнан)
---
# Протокол оценки (P0 — критический путь)

Связано: [[../tracks/AAAI-27]] · [[metrics]] · [[ablations]] · [[cohort-manifest]] · [[object-split-policy]] · [[../literature/roberts-2017]] · [[../literature/maurer-sanger-2023]]

## Текущий data/label-протокол (метод, anti-leakage)
- **Единая query-сетка `1…3000` для всех опытов** (uniform prediction horizon). `N_max`/`cycles_count`
  и фактическая длительность во входы НЕ подаются (в реальных опытах ≈ длительность → утечка).
- **Префикс** — причинные наблюдения до landmark-цикла N₀ (`prefix_landmark_cycles`), outcome-independent,
  без клиппинга по метке. Образцы с наблюдаемым (causal) пересечением в окне ≤ N₀ исключаются из risk set
  (`event_in_prefix`), а не переписываются.
- **Onset (событие/N_liq)** — первый цикл УСТОЙЧИВОГО пересечения ru≥0.95 на `onset_sustain_cycles`
  подряд идущих ЦЕЛЫХ циклах, на СЫРЫХ поцикловых пиках (поцикловое разрешение). Terminal-ambiguous
  (пересечение в 1–2 последних цикла без полного окна) исключаются опцией `exclude_terminal_ambiguous`.
- **Risk-метка по ОКНУ НАБЛЮДЕНИЯ:** `risk_label_observed` = разжижение ∨ (non-liq, доведённый до ≥H);
  non-liq, остановленные до H, — цензурированы и исключены из risk-метрик (без cure-предположения).
- **Event-time:** right-censored regression; `nliq_censor_valid` независим от risk-mask; regime-маски
  (liquefied / stabilized / unfinished) — по reached_horizon, отдельно.
- **Нагрузка:** измеренная CSR(N) из амплитуды девиатора; пропуски свойств — missingness-индикаторы.
- **Группировка:** site-held-out по `site_id` (см. [[object-split-policy]]).
- **Калибровка/coverage:** variance-scaling σ (НЕ conformal) + empirical site-held-out coverage с
  object-bootstrap CI (честная оценка, не finite-sample гарантия), ширина полосы рядом с покрытием.

## Почему это P0
Текущий headline — single-seed на **random split** → object leakage. Рецензент AAAI спишет метрики как inflated. Grouped CV реализован в ноутбуке **3_4** (`evaluation.cross_validation.make_grouped_cv_folds`).

## Целевой протокол
1. **Primary = object-held-out (grouped) CV.** Ни один объект не в train и test одновременно. random — только secondary «для сравнения с практикой». Обоснование: [[../literature/roberts-2017]].
2. **≥5 сидов, mean ± 95% CI** на КАЖДОЕ число headline-таблицы. Отчёт распределением, не «best seed by val» (= cherry-picking).
3. **Только 20 объектов** → grouped-фолды малы, CI широкие. Использовать **repeated grouped k-fold или LOO-object**; отчитывать **per-object variance**.
4. **Publication protocol, не demo epochs** — перезапустить с полными эпохами.

## Статистика значимости
- Per-sample ошибки (traj per-curve, |log N_liq err|) → **Wilcoxon signed-rank** (негауссовы) + **Holm–Bonferroni** (13 моделей).
- Неразложимые (AUROC/AUPRC/Brier/ECE/coverage) → **stratified bootstrap CI**, 1000 ресемплов.
- Везде **effect size + CI разницы**, не только p (на 1093 «значимо» бывает мизерным).

## Anti-leakage чеклист (red flags)
- [ ] **AUROC ≈ 1.0 — red flag.** На 14 landmark-eligible объектах это требует особенно сильного leakage/prefix-shortcut аудита.
- [ ] **Причинный префикс** — вход строится только из наблюдений ≤ N₀; никакого клиппинга по будущей метке (прежний клип лепил proxy-плато 0.949). Наблюдаемые onset'ы в окне → исключение из risk set, не переписывание.
- [ ] **DPI-Flow headline с `calibration_steps=0`** — inner-градиентная доводка θ не входит в нормированную плотность потока (шаги 1/2 — только абляция).
- [ ] no-prefix stress: если AUROC держится ~1.0 без префикса → искать утечку.
- [ ] OOD by soil / CSR / site.
