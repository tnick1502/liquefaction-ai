---
tags: [protocol, validation, P0]
status: TODO (multiseed не прогнан)
---
# Протокол оценки (P0 — критический путь)

Связано: [[../tracks/AAAI-27]] · [[metrics]] · [[ablations]] · [[../literature/roberts-2017]] · [[../literature/maurer-sanger-2023]]

## Почему это P0
Текущий headline — single-seed на **random split** → object leakage. Рецензент AAAI спишет метрики как inflated. `run_multiseed.py --grouped` существует, но **не прогнан** (`results/analysis_tables/multiseed_raw.csv` пуст).

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
- [ ] **AUROC ≈ 1.0 — red flag.** 0.9996 на 20 объектах = подозрение на leakage/prefix-shortcut.
- [ ] **Префикс строго ДО onset** — нет post-onset точек PPR во входном окне (иначе label leakage через вход).
- [ ] no-prefix stress: если AUROC держится ~1.0 без префикса → искать утечку.
- [ ] OOD by soil / CSR / site.
