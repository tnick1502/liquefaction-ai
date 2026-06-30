---
tags: [protocol, reviewer, fixes, code-review]
date: 2026-06-28
---
# Разбор критических замечаний рецензента и фиксы (по логике/коду)

Связано: [[p0-findings]] · [[object-split-policy]] · [[evaluation-protocol]] · [[metrics]]

> **Это история разработки (changelog фиксов), а НЕ методический референс и НЕ результаты-of-record.**
> Канонические методы — в [[evaluation-protocol]], [[metrics]], [[object-split-policy]], [[cohort-manifest]].
> Любые числа ниже — диагностические находки по ходу отладки (corr, доли, расхождения), они могут быть
> устаревшими; benchmark-результаты сюда не заносятся (см. политику в [[../00_INDEX]]).

| # | Замечание | Оценка | Что сделано |
|---|---|---|---|
| 1 | Leaderboard невалиден из-за устаревшего артефакта (config.json без prefix-флагов → существенная утечка) | **Верно** (операционно) | Код `config.py` уже содержит флаги; нужна **перематериализация** данных (1_x) + переобучение. Не кодовый баг — шаг запуска. Зафиксировано в [[p0-findings]] и README. |
| 2 | Primary CV не stratified: при страте <5 откат к KFold; фолды без CRR | **Верно, баг** | Переписан `make_grouped_cv_folds`: round-robin CRR-объектов первыми → **CRR-объект в каждом тест-фолде** при любых стратах. |
| 3 | Validation слабая: 1 CRR-объект, часто positive-only | **Верно** | `_pick_balanced_val`: val ≥2 объектов **обоих классов** + CRR-объект. Проверено на синтетике. |
| 4 | «conditional flow uncertainty» ≠ инференс (eps=0, один проход) | **Верно, критично** | Добавлен `DPIFlow.predictive(batch, K)` — **MC-пропагация через flow** (epistemic от θ + aleatoric от logvar-головы); `collect_outputs` зовёт его при `mc_samples_eval>1`. EVT-семейство — на гетероскедастичной голове (отмечено честно). |
| 5 | CV оценивает full-horizon Traj_RMSE, не post-prefix | **Верно** | `cross_validation` передаёт в P³ **continuation-метрики** (`Traj_RMSE_continuation_balanced/_worst`); p3_ranking их предпочитает. Headline-таблица уже на continuation. |
| 6 | Primary CV только 4 модели (нет сильных конкурентов) | **Верно** | `DEFAULT_MODELS` расширен: **Transformer, Neural Spline Flow, CatBoost, GRU, TCN** + флагманы + PINN. CatBoost через нативный `.fit` (не torch-цикл). |
| 7 | CI некорректны: 1.96·std/√5, фолды не независимы, псевдорепликация | **Верно** | Добавлен `object_cluster_bootstrap` (кластеры=объекты/площадки) на **pooled OOF**; для CI берётся один OOF-проход (без псевдорепликации repeats). Наивный CI помечен как справочная дисперсия. NB 3_5 использует cluster bootstrap как primary. |
| 8 | Runner неидемпотентен (`mode='a'` → дубли) | **Верно (для удалённого скрипта)** | В ноутбуках CSV пишутся **overwrite** (concat в памяти → один write). Добавлен `cv_grouped_run_meta.json` (prefix-флаг, seed, n_splits, n_repeats, mc). |
| 9 | CRR claim слишком широкий (мало проб/объектов) | **Верно** | Сплит теперь кладёт CRR-объект в каждый тест-фолд → CRR оценивается по нескольким объектам в CV. Claim формулировать как **exploratory CRR recovery** (см. [[../tracks/AAAI-27]] limitations). |
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
| 2 | калибровка не выполнялась в CV → покрытие занижено | `evaluate_fold` вызывает **fit_interval_scale на VAL каждого фолда** (torch-модели) |
| 3 | headline на наивном fold-CI | `headline_table(summary, cluster_df)` берёт **CI из object-cluster bootstrap**; NB 3_7 передаёт его |
| 4 | Wilcoxon псевдорепликация (тест по образцам, не по объектам) | `paired_significance(cluster='object')` по умолчанию: агрегат по объекту → тест по ~N объектов |
| 5 | нет no-prefix/no-aux stress | абляции **no_prefix / no_aux** через `stress_split` |
| 6 | абляции на 1 фолде | NB 3_6 default **FOLDS=[0,1,2]** |
| 7 | 2 теста красные | allowlist: добавлены `publication_headline_*.csv`; **PVR-тест оставлен строгим** (monotone_clip→0 by construction; падение = устаревший `full_leaderboard.csv`, позеленеет после перепрогона 3_1) |
| 8 | CRR не в primary CV | METRIC_KEYS += `CRR_RMSE, N_CRR_test, N_CRR_objects` |
| 9 | герой неочевиден | claim → Pareto-balance (см. [[../formulations/positioning-and-title]]) |
| 10 | MC: нет sens по K; CRR от последнего сэмпла | `predictive` теперь **усредняет CRR** по MC; K-sensitivity (4/8/16/32) — через `config.mc_samples_eval` (прогон-параметр) |

**Per-sample coverage90** добавлен в `compute_metrics` → cluster-bootstrap покрывает калибровку.

---

## Раунд N — два P0 протокола (data-leakage и conformal-гарантия)

> ⚠ **ЧАСТИЧНО SUPERSEDED разделом «Раунд N+1» ниже.** В частности: (1) «плановый горизонт
> `cycles_count`» оказался суррогатом длительности (≈last_obs) — заменён на ЕДИНУЮ сетку 1…3000;
> (2) «object-level CV+ … site-level **гарантия**» (строка про `object_loo_conformal_coverage`) была
> ранговой тавтологией — заменена на **empirical site-held-out coverage** (НЕ finite-sample гарантия),
> см. Раунд N+1 #3. Записи ниже сохранены для истории.
| # | Замечание | Фикс |
|---|---|---|
| 1 [P0] | Фактическая длительность опыта (`last_obs`) — endpoint входной сетки → утекает в risk/N_liq (длительность сильно коррелирует с N_liq) | `extract_test`: endpoint landmark-сетки = **ПЛАНОВЫЙ горизонт** (a-priori `cycles_count`, иначе глобальный `horizon_default`), **никогда** `last_obs`. `mask` валиден только до `last_obs` (доступность таргета, не длина входа). `load_object`/`build_real_objects_population` пробрасывают `horizon_default=max_cycle_reference`. Тест `test_grid_endpoint_is_planned_horizon_not_last_obs`. |
| 2 [P0] | conformal без гарантии: `val` используется и для early-stop, и для селекции, и для калибровки; per-specimen в 2–3 объектах | Введён **object-level CV+ conformal**: пер-траекторный nonconformity (`per_trajectory_nonconformity`) на TEST каждого фолда (модель НЕ видела объект) → агрегат до уровня объекта (макс по образцам) → LOO покрытие (`object_loo_conformal_coverage`, `aggregate_object_conformal`). Exchangeable unit = **объект/площадка** (site-level гарантия). Per-fold val-калибровка честно помечена как **диагностика** (NB 3_4 пишет `cv_object_conformal.csv`). |
| 3 [P1] | N_liq квантуется редкими поздними узлами сетки | Событие определяется на **поцикловой сглаженной огибающей** (`monotone_smooth(ppr_peaks)`) ДО ресэмплинга → поцикловое разрешение N_liq. Тест `test_nliq_per_cycle_resolution_and_event`. |
| 4 [P1] | conformal-блок в `except Exception: pass` → метрики молча исчезают | `evaluate_fold`: блок уважает `strict` (fatal / громкий WARN), как и калибровка. |
| 5 [P1] | Coverage публикуется без ширины полосы | `Coverage_90_splitconf_width` и `Coverage_90_simul_width` сохраняются рядом с покрытием (в `METRIC_KEYS`). |
| 6 [P1] | planned-горизонт и censor-time смешаны (non-liq N_liq=planned) | non-liq `N_liq` = **фактический last_obs** (right-censoring), planned — только для признака N_max/endpoint. Убрана импутация N_liq планом. Тест `test_nonliq_censor_is_last_obs_not_planned`. |
| 7 [P2] | synthetic-сетка зависит от global mutable `_LANDMARK_GRID` | `build_log_dense_cycles(..., landmark_n0, landmark_k)` — ЯВНЫЕ аргументы; global удалён. Order-independent (тест). |
| 8 [P2] | `fit_interval_scale` ошибочно назван conformal | Docstring/комментарии разведены: это **variance-scaling** калибровка σ (без конечновыборочной гарантии); честная оценка покрытия — **empirical site-held-out coverage** (Раунд N+1 #3), не formal guarantee. |

> **Важно:** изменения #1/#3/#6/#7 меняют материализацию данных — требуется **пересборка `data/dataset` + переобучение**, иначе старый артефакт несёт прежнюю утечку длительности.

---

## Раунд N+1 — глубокая ревизия (прежние правки были поверхностны; проверено на РЕАЛЬНЫХ данных)
| # | Замечание | Что реально не так было | Фикс (проверено) |
|---|---|---|---|
| 1/2 [P0] | Утечка длительности осталась: `cycles_count`≈`last_obs` (суррогат длительности), `N_max` — ВХОДНОЙ признак, сетка неоднородна | прошлый «a-priori horizon» = суррогат длительности; `log_cycle_norm`/`delta_cycle_norm` нормировались на N_max | **ЕДИНАЯ сетка 1…3000 для всех** (raw_loader endpoint=const); **N_max удалён из static features**; cyc-признаки нормируются на КОНСТАНТУ. Проверено на rebuild: endpoint единый для всех опытов, цикловые признаки не несут длительности, утечка устранена |
| 4 [P0] | DPI-Flow N_liq из ODE ДО residual (не лежит на финальной кривой) | nliq не пересчитывался после коррекции | пересчёт `soft_first_hitting` на ФИНАЛЬНОЙ кривой + **интерполяция порога** (N_liq лежит ТОЧНО на кривой). Проверено: N_liq теперь лежит точно на финальной кривой |
| 3 [P0] | «object-level CV+» — ранговая тавтология (давала ≈номинал при любых скорах) | сравнивал OOF-скоры сами с собой | удалён; **empirical site-held-out coverage**: q на VAL, покрытие на TEST-объектах + **object-bootstrap CI**. Тест доказывает не-тавтологичность (масштаб-зависимый q; покрытие отражает реальную долю) |
| 5 [P0] | Когорта смешана; ранняя цензура ошибочно входила в landmark risk set | фильтровались только ранние события | raw → strict landmark risk set: исключены ранние события и ранняя цензура до N₀; классы/сайты/CRR — в регенерируемом [[cohort-manifest]] |
| 6 [P1] | Одна маска для censoring, known-negative и physical regime | конфлейт | разведены `risk_label_observed`, `nliq_censor_valid`, trajectory/continuation и `regime_stable/unfinished`; все потребители читают явную маску |
| 7 [P1] | CRPS/NLL/coverage/conformal на ПОЛНОЙ маске | префикс тривиально покрыт → завышенная калибровка | переведены на **post-prefix (continuation) маску** (метрики + conformal-скоры в CV) |
| 8 [P1] | Reliability не фильтрует незавершённые non-liq | ECE расходился с таблицей | `reliability_diagram` фильтрует по `n_liq_observed` |
| 9 [P1] | CatBoost: censored N_liq как точный RMSE-таргет | несимметрично с censored loss proposed-моделей | регрессор учится только на exact events; censored-aware N_liq и P³ помечаются N/A, отдельно публикуется liquefied-only metric |
| 10 [P2] | n_folds=0 у CatBoost; wo_conformal глотал ошибку; опыт 500 циклов «короткий» | count по ненулевому P3; `except: pass`; undershoot геом-сетки | n_folds по размеру группы; ошибка калибровки не глотается; `_terminal_observability` берёт ФАКТический last_obs (n_liq), не узел сетки |

> Всё data/loss/feature-затрагивающее (1,2,3,6,7,9) вступит в силу после **пересборки `data/dataset` (33 признака) + переобучения всех моделей**. Старые веса 34-мерные и несовместимы с новой постановкой.

---

## Раунд N+3 — нагрузка, пропуски, survival/монотонность
| # | Замечание | Решение |
|---|---|---|
| 9 [P1] | object-held-out ≠ geographic site-held-out (ВГК-5/ВГК-6 один адрес, разные id) | `canonical_site_id(oname)` (адрес-ключ), колонка `site_id` в meta, ВСЯ группировка CV по `site_id` (`_group_col`). Проверено: ВГК-5/6 сворачиваются в одну площадку; тест запрещает их разъезд по фолдам. |
| 4 [P0] | P³ states через хрупкую плато-эвристику (`rise_eps`) | `regime_stable/unfinished` переопределены по ОКНУ НАБЛЮДЕНИЯ (reached_horizon), eps-независимо; одно определение для P³-метрик и trigger-suppression моделей. |
| 3 [P0] | CSR — константа, статья обещает `CSR_i(N_t)`; шторм = ручной `nonstationarity=0.30` | Путь (а): **измеренная CSR(N) из амплитуды девиатора** (`extract_cycle_amplitude`), привязана к `CSR_base`; `nonstationarity` теперь **data-derived** (CV амплитуды), не зашитая. **Честный вывод данных:** опыты контролируемо-амплитудные (амплитуда девиатора почти постоянна) — CSR(N) выходит ~плоской; «штормовая нестационарность» в сыром сигнале ОТСУТСТВУЕТ (это категориальный протокол, не амплитудная программа). Теперь это видно из данных, а не выдумано. |
| 10 [P1] | пропуски → константы (`e=0.7`,`G=25`,…), статья «rather than fabricated defaults» неверна | Добавлены **missingness-индикаторы** `miss_e/miss_Ip/miss_K0/miss_vs/miss_gran` в статические признаки. Аудит на реальных данных вскрыл, что **Vs почти всегда из fallback** (G=25), а у значительной части опытов нет гранулометрии — индикаторы честно раскрывают суррогатность входов. |
| 7 [P1] | censor-aware regression ≠ proper survival (нет AFT/hazard, IPCW/IBS/time-dependent AUC) | **Принято как limitation** (по решению): формулировка скромная — «right-censored event-time regression», БЕЗ survival-claim. Информативная цензура (разные протоколы остановки) и отсутствие IPCW/IBS — явный пункт limitations. Полный survival-объектив — future work. |
| 13 [P2] | глобальная изотоника таргета использует будущее → PVR частично проверяет навязанное | **Принято как limitation** (по решению): глобальная монотонизация PPR-таргета — предположение; есть `causal_monotone_smooth` (поцикловой causal вариант, не использует будущее) для аудита (расхождение global vs causal по меткам мало). PVR трактуется как «соответствие архитектурному monotone-проекту», не независимая физическая истина. |
