# DPI-Flow → AAAI-27: сводные рекомендации

> Объединяет `recomendation.md` (критика протокола/статистики) и `рекомендации.docx`
> (позиционирование, структура, дедлайны) + критические дополнения, которых не было ни в одном из двух.
> Версия: 2026-06-28. Заменяет оба исходника. Связанная база знаний: `knowledge_base/`.

---

## 0. Статус и жёсткие сроки AAAI-27

- **Формат:** Main Technical Track, **7 страниц** technical content + неограниченные references.
- **Дедлайны (CFP AAAI-27):** abstract — **21 июля 2026**, full paper — **28 июля 2026**, supplementary/code — **31 июля 2026**.
- **Критический путь, который сейчас недооценён:** сам `article/AuthorKit27/AnonymousSubmission2027.tex` — **пустой шаблон**, `results/analysis_tables/multiseed_raw.csv` — **пуст**. То есть ни текста, ни статистики ещё нет. Инфраструктура (`run_multiseed.py`, флаг `GROUPED`) написана, но не прогнана.
- **Реальный объём работы до подачи** = (а) прогнать статистику, (б) дозакрыть абляции, (в) написать 7 страниц с нуля. Планировать надо от этого, а не от «моделей не хватает» — моделей уже избыточно.

---

## 1. Вердикт и позиционирование

**План с DPI-Flow как главным героем — правильный.** AAAI-статья = **один чистый ML-вклад**: conditional flow над физически ограниченной дифференцируемой динамикой для censored onset forecasting. DPI-EVT / EVT-NeuralSSM — baselines/extensions; topology — appendix/future work; геотехника — ровно столько, чтобы доказать, что постановка реальная и сложная.

**Позиционирующая фраза:** *DPI-Flow is a physics-structured probabilistic latent-ODE/flow architecture for prefix-conditioned event-time forecasting under censoring and site-level distribution shift.*

**Title (рекомендация):** `Physics-Constrained Conditional Flows for Liquefaction Onset Forecasting`.
Слово **Calibrated в title — рано**: Coverage@90 = 0.975 это скорее *conservative*, а не идеально калиброванная неопределённость. Calibration заявлять как раздел/результат, не как title-claim, пока нет сильной reliability story (см. P2).

**Главный claim переформулировать честно.** Сырой Traj_RMSE/CRPS сейчас лучший у **Neural Spline Flow** (RMSE 0.1029, CRPS 0.0542), но у неё Physics_Violation_Rate = 0.917. Поэтому claim — **не** «best predictor overall», а:
> **best physically admissible, onset-aware probabilistic forecaster under object-held-out evaluation.**

**Почему DPI-Flow — герой (фактура):** лучший admissible P³ (524.24); лучший N_liq_logMAE среди trajectory+physics (0.164); лучший N_liq_MAE среди всех trajectory-моделей (38.9 циклов); нулевой physics violation rate; единственное семейство, дающее CRR + trajectory + risk + uncertainty + latent physical θ.

---

## 2. P0 — что убьёт статью (делать первым)

### 2.1 Object leakage (самое опасное)
Headline посчитан на **random split** → образцы одного объекта в train и test. Рецензент спишет метрики как inflated.
- **Primary-протокол = object-held-out (grouped) CV**, random — только secondary «для сравнения с прошлой практикой».
- `run_multiseed.py --grouped` существует, но **не прогнан**. Прогнать.

### 2.2 AUROC ≈ 1.0 — это RED FLAG, не достижение
AUROC около 1.0 на 14 landmark-eligible объектах рецензент прочтёт как leakage или prefix-shortcut.
- **Обязательно проверить, что префикс строго ДО onset** (нет post-onset точек PPR в наблюдаемом окне). Если есть — это label leakage через вход, всё рушится.
- Закрыть стресс-тестами: **no-prefix**, **prefix-length sweep**, continuation-only метрики, object-held-out. Если AUROC падает под стрессом — это правильно и честно; если держится ~1.0 без префикса — ищите утечку.

### 2.3 Single-seed + best-of-seeds = cherry-picking
Финальные флагманы выбираются «best seed by val», модели seed-sensitive.
- Нужно: **mean ± 95% CI по ≥5 сидам** для КАЖДОГО числа headline-таблицы. Отчёт — распределением, не лучшим прогоном.
- Из-за **20 объектов** grouped-фолды малы → CI широкие. Использовать **repeated grouped k-fold или LOO-object** для получения распределения и per-object variance.

### 2.4 Нет тестов значимости
- Per-sample ошибки (per-curve traj error, per-sample |log N_liq err|) → **Wilcoxon signed-rank** (ошибки негауссовы) + **Holm–Bonferroni** (13 моделей → множественные сравнения).
- Неразложимые метрики (AUROC/AUPRC/Brier/ECE/coverage) → **stratified bootstrap CI**, 1000 ресемплов по тест-образцам.
- Везде давать **effect size + CI разницы**, не только p. На 790 benchmark-образцах «значимо» бывает мизерным — показать честно.

> P0 — первым, потому что grouped+multiseed может **перетасовать лидерборд**. Абляции и текст до этого = риск переписывать.

---

## 3. P1 — абляции (1:1 к заявленным вкладам)

Уже есть (notebook 3_2): w/o flow, w/o ODE, NeuralODE w/o physics. **Мало.** Добавить:

1. **w/o conformal calibration** — Coverage@90 + ECE до/после. Прямая опора claim «calibrated». Обязательная строка.
2. **Flow-posterior vs честный Gaussian-posterior над θ** (не просто `use_flow` on/off). Рецензенты подозревают, что flow — overkill; показать, что экспрессивность нужна, иначе резать.
3. **w/o monotonicity projection** (`monotone_clip`) — рост Physics_Violation_Rate и реальный trade-off с RMSE.
4. **w/o discriminative risk head / soft-AUC** (`risk_clf`, `prior_gate`) — влияние на AUROC/AUPRC.
5. **Censoring/Tobit для N_liq** (with/without одностороннего цензурированного loss) — цензура при N_max критична; без абляции N_liq-метрика под вопросом.
6. **Prefix-length sensitivity** — AUROC и N_liq-ошибка vs доля наблюдённого префикса (10/20/30/50%). Абляция + killer-фигура для early-warning рамки.
7. **Robustness к пропускам Vs/grainsize** — реальный Vs лишь у 16.7%, grainsize у ~55%. Без стресс-теста на imputation валидность фич атакуема.

**Где:** свести в одну Table 3 с mean±CI и **разбивкой по 3 состояниям** (liq/stab/nostab). Каждую абляцию — на grouped split + multiseed.

---

## 4. P2 — метрики (фокус, не количество)

- **Onset:** вперёд **AUPRC** на известных by-3000 исходах (**460 pos / 263 neg**; остальные 67 цензурированы до H), а не AUROC; Brier; ECE + reliability diagram с `risk_label_observed`.
- **Lead-time / timeliness:** за сколько циклов до фактического разжижения модель поднимает риск → осмысленная early-warning метрика.
- **N_liq:** headline-метрику считать **только на нецензурированных** (или censored-aware), явно оговорив. MAE по right-censored таргетам легко атаковать.
- **Trajectory:** вперёд **CRPS + NLL** (proper scoring), а не RMSE — вклад вероятностный. Плюс **PICP/MPIW** (coverage + ширина интервала) вместе.
- **Per-state (3-regime)** как primary; pooled — вторично. `Traj_RMSE_worst` — robustness-история.
- **CRR recovery RMSE** — уникальная способность DPI-семейства, **secondary** (CRR-caveat: 5 объектов / 349 benchmark-образцов; claim осторожный).

---

## 5. Структура 7-страничной статьи + фигуры/таблицы

| Секция | Объём | Содержание |
|---|---|---|
| 1. Introduction | 0.75 | Onset важен; CSR/CRR интерпретируемы, но не prefix-conditioned probabilistic; чисто нейронные нарушают физику; DPI-Flow. |
| 2. Problem Setup | 0.75 | Inputs: soil + CSR history + early PPR prefix. Outputs: PPR, risk, N_liq, CRR. Три режима цензуры. Object-held-out. Метрики: post-prefix RMSE primary. |
| 3. Method: DPI-Flow | 1.5 | Encoder; conditional coupling RealNVP над θ; analytical differentiable ODE-layer; exact-forward/soft-backward first-hitting; `L = L_traj + L_risk + L_censored_Nliq + L_aux + L_KL/flow + L_physics`; variance scaling + empirical held-out coverage audit. |
| 4. Experimental Setup | 1.0 | raw=1093 / 20 объектов → landmark risk set **790 / 14 объектов** после исключения 97 ранних событий и 206 ранних цензур; **460 liq / 330 non-liq**; risk-known 460/263; seismic=436 / storm=354; **CRR 349 образцов / 5 объектов**. Единый горизонт 1…3000; три физические regime-маски отделены от censoring. |
| 5. Results | 1.5 | Main table (все модели); отдельная admissible table (physics-feasible); calibration/reliability plot; post-prefix trajectory case study; P³ как **secondary engineering ranking**. |
| 6. Ablations & Robustness | 1.0 | 7 абляций из §3; object/LOO CIs. |
| 7. Limitations & Conclusion | 0.5 | CRR измерена на 5 объектах / 349 benchmark-образцах; physics = assumptions; prefix упрощает задачу; topology — future work. |

**Фигуры main:** (1) method diagram prefix+soil→encoder→flow θ→analytical layer→PPR/risk/N_liq/CRR; (2) compact leaderboard; (3) reliability + coverage table; (4) ablation table; (5) case-study trajectory с uncertainty band; (6) опционально latent θ UMAP/Mapper как маленькая interpretability-панель.
**Supplement:** full leaderboard; OOD by soil/CSR; censoring details; reproducibility checklist; topology branch; extra case studies; полные гиперпараметры.

**Contribution bullets (AAAI-стиль):**
1. Prefix-conditioned onset-forecasting formulation для cyclic liquefaction tests.
2. Conditional parameter flow + analytical differentiable liquefaction layer (feasible monotone accumulation).
3. Censored onset objective: liquefied exact; каждый landmark-eligible non-liq right-censored на фактическом last_obs; stabilized/unfinished анализируются отдельно.
4. Object-held-out benchmark: raw=1093/20 объектов → landmark risk set **790/14 объектов** (460/330), со strong baselines.
5. Transparent uncertainty/physics evaluation (post-prefix RMSE, censored N_liq, calibration, physics violations, CRR recovery).

---

## 6. Литература и baselines

Полная карта цитирования, atomic-заметки на каждую работу с DOI, claim→citation и минимальный список — в **`knowledge_base/`** (см. `knowledge_base/00_INDEX.md`).

**Минимум, чтобы рецензент не снёс по baselines:** state-of-practice geotech (Seed&Idriss, Boulanger&Idriss, Cetin/Moss/Kayen) + GBDT (CatBoost/XGBoost) + deep tabular (FT-Transformer/TabNet) + sequence (GRU/TCN/Transformer) + survival/event-time (Cox/DeepSurv) + **хотя бы одна опубликованная liquefaction-ML / seq-forecast SOTA со ссылкой** (иначе «все бейзлайны самодельные»). Ближайший конкурент по духу — Sanger, Geyin & Maurer (2025), mechanics-informed ML.

---

## 7. Must-do чеклист до подачи (приоритизированный)

- [ ] **P0-a** Прогнать `run_multiseed.py --grouped` и random, ≥5 сидов → таблица mean±95%CI.
- [ ] **P0-b** Wilcoxon+Holm (per-sample) + bootstrap CI (неразложимые) + effect sizes.
- [ ] **P0-c** Проверить, что **префикс строго до onset** (нет post-onset утечки) — иначе AUROC≈1 невалиден.
- [ ] **P0-d** Перезапустить **publication protocol, не demo epochs** (часть текущих результатов выглядит «быстрыми»).
- [ ] **P1** 7 абляций на grouped+multiseed → Table 3 с per-state разбивкой.
- [ ] **P2** Reliability diagram, prefix-length curve, lead-time, censored-aware N_liq, CRPS/NLL/PICP/MPIW вперёд.
- [ ] **CRR** Либо stronger split/report, либо честная формулировка «CRR recovery is auxiliary, limited by held-out CRR object count (N_CRR_objects=1)».
- [ ] **Topology** Убрать из main claims полностью (→ supplement/future work).
- [ ] **Claims** Переписать так, чтобы NSF (лучший raw RMSE) не ломал нарратив → «best physically admissible onset forecaster».
- [ ] **Repro** Заполнить `ReproducibilityChecklist.tex` (seeds, splits, configs, model cards, dataset schema, code release plan) — AAAI это требует.
- [ ] **Текст** Написать 7 страниц с нуля под методную рамку (`.tex` сейчас шаблон).

---

## 8. Что я добавил критически (не было ни в одном из двух документов)

1. **`.tex` — пустой шаблон + multiseed.csv пуст.** Это и есть реальный критический путь, а не «доработка моделей».
2. **AUROC≈1.0 как red flag и проверка prefix-leakage** (post-onset точки во входе). Документ-рекомендации этого не ловил; это первый вопрос рецензента.
3. **Малая выборка по объектам (20):** grouped-фолды малы → нужен **repeated grouped k-fold / LOO-object** и отчёт per-object variance, иначе CI неинформативны.
4. **Multiple-comparison correction** при 13 моделях (Holm-Bonferroni) — обязательна, иначе p-значения невалидны.
5. **CRR практически невалидирован held-out** (N_CRR_objects=1 в test): либо отдельный CRR-split/LOO, либо явный downgrade до auxiliary в limitations.
6. **Reproducibility checklist как блокер подачи** (файл есть, но пустой шаблон).
7. **Effect size > significance** на малых данных: «значимо, но мизерно» надо показывать явно.

---

## 9. Порядок работы

**P0** (статистика + leakage-проверка) → **P1** (абляции) → **P2** (метрики/фигуры) → **текст** под методную рамку → **repro checklist**. AAAI first; workshop (topology) и журнал — после, не смешивать с AAAI (см. `knowledge_base/tracks/`).
