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

---

## Раунд N+2 — правки по замечаниям (только логика, ноутбуки не перезапускались)
| # | Замечание | Оценка | Фикс |
|---|---|---|---|
| P0 | Сплит по `site_id`, но статистика/conformal/significance кластеризуют по `object` (2 скважины одной площадки = 2 «независимых» кластера) | **Верно** | Кластер везде → **`site_id`** (fallback object): `_SAMPLE_COLS`+`compute_metrics` пробрасывают `site_id`; conformal val-калибровка и `aggregate_object_conformal` группируют по site_id; `significance.paired_significance`/`object_cluster_bootstrap` → site_id. Проверено: n падает с 12 скважин до 6 площадок. |
| P0 | 3_1 ROC/calibration/temperature на всех 229 метках (вкл. 9 unknown-by-H), а leaderboard — на 220 наблюдаемых | **Верно** | Ячейки 15/17/19 фильтруют по `risk_label_observed` (та же маска, что `compute_metrics`). |
| P0 | Тест падает: DPI-Flow PVR 0.00873 из-за `clamp→1.0500001 > 1.05` | **Верно (числ.)** | `metrics.py`: проверка границы с допуском `1e-4` (`pred>1.05+eps`). Реальных нарушений нет → структурные модели снова PVR=0. |
| P1 | `wo_monotone` = no-op (побитово = full): `monotone_residual_scale` делает кривую монотонной независимо от флага | **Верно** | При `use_monotone_clip=False` residual теперь **обычный аддитивный** (`+0.1·tanh`), проекция = bound-clamp → абляция реально измеряет вклад монотонности. |
| P1 | Гиперпараметры выбираются по full `Traj_RMSE`, не continuation | **Верно** | 2_2/2_3/2_4: `SELECTION_METRIC="Traj_RMSE_continuation"`. |
| P1 | `cv_grouped_run_meta.json` пишется до CV → ложная готовность | **Верно** | Запись перенесена в конец CV-ячейки (после summary), с `n_folds_done`. |
| P2 | 3_2 — 2 эпохи, single-class OOD (AUROC NaN); вывод «структурные устойчивее» не подтверждён | **Верно** | Эпохи → publication; добавлен banner «exploratory, superseded by 3_6/3_4, conclusions diagnostic only». |
| P2 | 3_3: кейсы-экстремумы, measured за концом опыта, «posteriors» = кросс-образцовые point estimates | **Верно** | Выбор → **репрезентативные** (медиана PPR_max); measured маскируется `valid_mask`; заголовок гистограммы честно переименован. |

**Осталось (операционно, не код):** прогнать по порядку 3_1→3_3→3_4→3_5→3_6→3_7→3_8 (тогда 3_7/3_8 перестанут быть устаревшими, leaderboard будет на CV, а не на 2 test-site). Числа абляции prefix/onset в статье обновить ПОСЛЕ прогона (сейчас prefix слабо влияет на onset → honest reframe: onset ведут статические/site-признаки, prefix помогает траектории/N_liq).

---

## Раунд N+3 — инженерия кастомных моделей (N_liq EVT + калибровка DPI-Flow)

**Диагностика (per-sample CV):** N_liq EVT-семейства в object-held-out CV — НОРМА (log-MAE на liq: DPI-EVT 0.31, EVT-SSM 0.31, DPI-Flow 0.34). Значение **0.84/1.7** — только на **single-split из 2 площадок** (3_1), не на CV. Причина хрупкости: N_liq брался лишь из пересечения кривой; при непересечении остаточная масса → горизонт 3000 → катастрофа на OOD.

| Задача | Решение (выбор автора) | Код |
|---|---|---|
| N_liq EVT/DPI-EVT | **Head + consistency + report CV** | Добавлена выделенная `nliq_head` (MLP, флаг `use_nliq_head=True`, backward-compat через `**kwargs`) в базовый `EVTNeuralSSM` (наследует DPI-EVT). `_apply_nliq_head` даёт primary N_liq из головы; `_nliq_consistency_loss` (вес 0.10) тянет голову к пересечению кривой ТАМ, где событие произошло (label==1 & observed). `nliq_norm_curve` в выходах. Headline/competence-gate брать из **CV (3_4)**, single-split (3_1) — диагностика. |
| DPI-Flow калибровка | **Conformal (headline)** | Headline-калибровка = **empirical site-held-out conformal coverage** (`aggregate_object_conformal`, уже в 3_4; выведена в 3_7). ~~Variance-scaling запас 1.15~~ → **откачен к 1.0 в Раунде N+4** (был test-informed). Никакой подгонки; полоса под site-shift выбирается только внутри outer-train, если нужна. |

**⚠ Требует переобучения `2_3` (EVT-SSM) и `2_4` (DPI-EVT):** добавлен параметр `nliq_head` в архитектуру → старые `weights.pt` не загрузятся strict-режимом в 3_1 до перетренировки. Порядок: 2_3→2_4→3_1→3_3→3_4→…→3_8.
**Честная оговорка для статьи:** N_liq теперь из головы, регуляризованной к физическому пересечению кривой (не строго лежит на кривой) — указать в methods/limitations.

---

## Раунд N+4 — критические правки (6 P0 + P1, только логика)

| # | Замечание | Оценка | Фикс |
|---|---|---|---|
| 1 | **Prefix и event-label противоречат:** событие берётся из полной изотонической огибающей, префикс — из отдельного сглаживания раннего окна → 13 отрицательных с prefix≥0.95 и 4 положительных с onset внутри N₀ | **Верно** | `sustained_first_crossing` (порог удержан на `onset_sustain_cycles=3` **подряд** циклах) применяется к **СЫРЫМ** пикам для метки/N_liq (ретроспективный таргет) и к **сырым пикам причинного окна ≤ N₀** для причинного флага. **НИКАКОГО клиппинга входа** (прежний `min(prefix, порог)` был outcome-conditioned и лепил proxy-плато 0.949 — удалён). Причинный `event_in_prefix` (label-free) исключает образец из landmark risk set; расхождения причинного и ретро-критериев **аудируются счётчиками** (`audit_*`), не «чинятся». |
| 2 | Когорта в config названа «~19/19», а по аудиту 1093/20 raw → 826/13 landmark | **Верно** | Комментарий config исправлен: raw **1093 опытов/20 объектов → landmark 826/14 объектов/13 площадок** (исключено 267/6). В статье — обе цифры. |
| 3 | **Плотность потока некорректна после inner-calibration:** `calibration_steps>0` делает градиентные шаги по θ, якобиан которых НЕ входит в нормированную плотность → mixture-NLL невалиден | **Верно** | Headline DPI-Flow: **`calibration_steps=0`** (честная плотность); грид 2_2 → `[0]`. Эвристическая θ-доводка (1/2 шага) вынесена в **абляции** `calib_steps_1/calib_steps_2` (`ablation_study`, conformal-полоса, а не плотность). |
| 4 | N_liq consistency может тянуть голову к горизонту на непересекающих кривых | **Верно** | `_nliq_consistency_loss`: target **detached**, вес гейтится `cross_mass≥0.5` (только где кривая реально пересекла) × `label` × `observed`. |
| 5 | `calibration_shift_inflation=1.15` — test-informed подгонка (введена после наблюдения held-out недокрытия) | **Верно** | Откат к **1.0** (никакой подгонки). Honest headline = empirical site-held-out conformal coverage. Мислейбл «conformal s:» в 2_2/2_3 → «var-scale s:»; ablation `wo_conformal`→**`wo_varscale`**. |
| 6 | Старые веса × новая арх.; preflight проверяет только static_dim; нет манифеста прогона | **Верно** | `test_scientific_claims`: preflight сверяет **полный state_dict (ключи+формы)** сконструированной из hyperparams модели с `weights.pt`. Добавлен **run manifest** (`evaluation/manifest.py` → `results/run_manifest.json`): git-commit(+dirty), config, SHA1-отпечаток данных, состав когорты, архитектуры; ячейка в 3_1. |
| P1 | A/B `train_ab_pair` кластеризует по object, полная маска, single-split | **Верно** | `ab_test`: скоры на **строго post-prefix continuation**, кластер bootstrap по **site_id**, добавлена A/B-калибровка **`Cov90_abs_miscal`** (\|cov90−0.90\|), multi-fold `ab_flow_vs_gaussian_pooled` (пул per-sample по фолдам → один site-bootstrap). |
| P1 | CRR-diversity, site-macro | **Верно** | `N_CRR_objects` считает **site_id**; добавлены **site-macro** `Traj_RMSE_continuation_siteMacro`, `N_liq_logMAE_siteMacro`, `N_sites_test` (каждая площадка весит одинаково — не доминируется крупными). |

### Reframing (только текст статьи/доки, кода не требует)
- **MC-mixture (`MC_SOTA`)** — опция, а не headline: заявлять как отдельный SOTA-эксперимент, не смешивать с основным сравнением.
- **θ-uncertainty — условный латент, НЕ байесовская эпистемика:** разброс θ идёт из амортизированного гауссова/flow-постериора по входу, а не из апостериора по параметрам модели. В methods писать «conditional posterior over physical params», не «epistemic Bayesian uncertainty».
- **N_liq head детерминирована** — при этом модель заявляет вероятностный прогноз: указать, что вероятностной является траектория PPR, а точечная N_liq выводится из головы (+интервал из континуальной кривой), это разные величины.
- **`Onset_EarlyWarning_Rate` геймабелен** (можно занизить порог) — сопровождать caveat и парой с ложной тревогой; не headline-метрика.
- **θ-интерпретируемость** — заявлять только с тестом восстановления (identifiability), иначе как «структурный индуктивный bias», не «интерпретируемые физпараметры».
- **Численная сходимость ODE** — добавить контроль Euler vs Heun (`integrator` уже в гриде EVT) как sanity, а не точность.
- **Survival-baseline** формулировать как «right-censored regression на N_liq», честно называть цензуру.

**⚠ Требует перегенерации артефакта + переобучения (арх./данные менялись):** новый sustained-onset и причинный risk-set меняют данные → `data/prepare_dataset.ipynb` заново, затем 2_1..2_4, затем 3_1..3_8. Числа prefix/onset и абляций в статье обновить ПОСЛЕ прогона.

---

## Раунд N+5 — исправление ОШИБОК в раунде N+4 (по повторному критическому ревью)

Четыре пункта раунда N+4 были сделаны неверно/недостаточно; исправлено:

1. **Клип префикса был outcome-conditioned** (`grid < n_liq`, использовал будущую метку) и создавал proxy-плато ровно `0.949`. **Удалён.** Причинный префикс строится только из наблюдений ≤ N₀; образцы, где причинный префикс уже устойчиво пересёк порог, **исключаются** из landmark risk set через label-free `event_in_prefix`, а НЕ переписываются. Противоречия причинного и ретроспективного критериев считаются отдельно (`cohort_filter_counts.audit_*`).
2. **Sustained-onset фактически не работал:** (а) усечённое окно `above[i:min(i+sustain,n)]` пропускало 1–2 хвостовые записи; (б) критерий применялся к УЖЕ монотонной `ppr_sm_pc`, где после первого пересечения кривая по построению ≥ порога → sustain был **no-op**. Теперь: функция вынесена как `sustained_first_crossing`, применяется к **сырым `ppr_peaks`**, требует **полное окно** из 3 подряд, с корректной обработкой терминального onset (≥2 последних цикла). Юнит-тесты: `tests/test_onset_criterion.py`.
3. **Терминология:** метка/N_liq берутся из глобальной **некаузальной** `monotone_smooth` — это допустимо для ретроспективного ТАРГЕТА, но называть её «причинной» нельзя. В доке/статье: причинным является **только префикс** (вход); метка — retrospective target. Формулировки исправлены.
4. **Primary nested CV всё ещё перебирал density-invalid DPI-Flow:** `cross_validation.NESTED_GRIDS["dpi_flow"]` имел `calibration_steps:[1,2]` (а фикс был только в одиночном 2_2). → **`calibration_steps` убран из грида (фиксирован 0)** для primary DPI-Flow; `dpi_evt` тоже → `[0]`. Шаги 1/2 остаются ТОЛЬКО абляцией (`ablation_study: calib_steps_1/2`), не в headline-density.

---

## Раунд N+6 — pre-final-run review (критическая проверка правок автора)

Правки автора (consecutive-cycle sustain, terminal-ambiguous audit, расширенный manifest) — верное усиление. Найдено и исправлено перед финальным прогоном:

1. **Скрытый баг маскировки onset в режиме `points_in_cycle`:** проверка «подряд идущих циклов» сравнивала СЫРЫЕ float-номера циклов пиков (`|Δ−1|≤0.25`). У реальных данных пики берутся как argmax внутри окна цикла, и около onset фаза пика дрожит: два СОСЕДНИХ цикла дают номера вида `k+0.9` и `(k+1)+0.1` → Δ≈0.2 → критерий **ложно отвергал бы реальный onset и цензурировал разжижение** ровно там, где волна искажается. → Последовательность теперь определяется по ЦЕЛОМУ номеру цикла `floor(cyc+ε)` (`_cycle_bins`/`_consecutive_cycles`), устойчиво к субцикловому дрожанию; настоящие пропуски циклов по-прежнему отвергаются. Тест `test_subcycle_phase_jitter_still_consecutive`. **Критично: без этого фикса финальный прогон в points_in_cycle-режиме занизил бы число позитивов.**
2. **Терминально-неоднозначные образцы** (пересечение в последние 1–2 цикла, полного окна нет) держались как censored-негативы → шум метки на позитивном классе. → Конфиг `exclude_terminal_ambiguous=True` исключает их из размеченной когорты (флаг `onset_terminal_ambiguous` остаётся в meta; счётчики `terminal_onset_ambiguous`/`excluded_terminal_ambiguous`). Поставить False для sensitivity.
3. **Stale-комментарий** в `extract_test` всё ещё описывал приём терминального onset как события (старое поведение) → синхронизирован с кодом (теперь — terminal-ambiguous audit).
4. **Проверки утечки признаков:** подтверждено — `event_in_prefix`/`onset_terminal_ambiguous` снимаются из `load_rows` ДО `load_df`; `static_features` — явный allowlist, новая meta-колонка `onset_terminal_ambiguous` НЕ попадает во входы. `validate_run_manifest` экспортирован как publication-gate.

**Проверка перед прогоном (обязательно посмотреть после `prepare_dataset`):** распечатать `population["cohort_filter_counts"]` — особенно `terminal_onset_ambiguous`, `excluded_terminal_ambiguous`, `audit_*`. Если `terminal_onset_ambiguous` велик (десятки), это сигнал, что запись опытов обрывается слишком близко к onset — задокументировать в статье и проверить чувствительность (`exclude_terminal_ambiguous=False`).
