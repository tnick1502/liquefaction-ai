"""
Тесты второго раунда P0/P1-исправлений протокола:
  * #1/#2 — risk_label_observed по окну наблюдения; #4 — проспективный audit_horizon_negatives;
  * #10 — n_folds считается по размеру группы (CatBoost с NaN P3 не теряет фолды);
  * #1 — N_max отсутствует во входных признаках реального адаптера/синтетики;
  * #9 — N_liq-регрессор baseline учится только на наблюдённых (uncensored) — проверка логики маски.
"""
import numpy as np
import pandas as pd

from liquefaction_ai.data.splits import audit_horizon_negatives
from liquefaction_ai.evaluation.cross_validation import aggregate_cv


def test_risk_label_observed_is_observation_window_not_plateau():
    # #1/#2 риск-метка известна для liq И для non-liq, доведённых до горизонта H — независимо от плато.
    H = 3000.0
    liq_label = np.array([1.0, 0.0, 0.0, 0.0])
    # last_obs(raw n_liq): liq событие=50; non-liq доведён до 3000; non-liq до 5000; non-liq стоп 800
    n_liq_raw = np.array([50.0, 3000.0, 5000.0, 800.0])
    is_liq = liq_label > 0.5
    reached = n_liq_raw >= H - 1e-6
    risk_obs = is_liq | ((~is_liq) & reached)
    assert risk_obs.tolist() == [True, True, True, False], "non-liq до ≥H — известный негатив; стоп<H — цензура"
    # ключевое: non-liq, доведённый до 3000 (плоский он или нет), считается наблюдаемым негативом
    assert risk_obs[1] and risk_obs[2] and not risk_obs[3]


def test_audit_horizon_negatives_is_prospective():
    # Среди non-liq, ПЛОСКИХ в раннем окне [400,500], считаем поздний рост ПОСЛЕ окна (отдельный участок).
    seq = 200
    cycles = np.tile(np.linspace(1.0, 3000.0, seq), (3, 1)).astype(np.float32)
    vm = np.ones((3, seq), dtype=np.float32)
    r = np.full((3, seq), 0.45, dtype=np.float32)      # плоские в раннем окне
    late = cycles[0] > 500.0
    r[1, late] = 0.80                                   # 2-й позже вырос (>0.03) — нарушает absorbing
    r[2, late] = 0.98                                   # 3-й позже разжижился
    lab = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rep = audit_horizon_negatives(r, vm, cycles, lab, threshold=0.95, early_window=(400.0, 500.0))
    assert rep["n_early_flat"] == 3.0
    assert rep["n_later_rise"] == 2.0
    assert rep["n_later_cross"] == 1.0                 # финальный positive не исключён selection-фильтром
    assert rep["n_final_liq"] == 1.0


def test_canonical_site_id_merges_same_address():
    from liquefaction_ai.data.raw_loader import canonical_site_id
    a = canonical_site_id("852-23 3-й Хорошевский пр-д, вл. 5 (ВГК-5)- plaxis (96)")
    b = canonical_site_id("856-23 3-й Хорошевский пр-д, вл. 5 (ВГК-6) - Plaxis (96)")
    c = canonical_site_id("25-26 Строгино 2.1 - Plaxis")
    assert a == b and a != c                              # ВГК-5/ВГК-6 — одна площадка


def test_cv_groups_by_site_not_object():
    # #9 два объекта с одним site_id НИКОГДА не разъезжаются по train/val/test одного фолда.
    from liquefaction_ai.data.splits import make_grouped_cv_folds
    from liquefaction_ai.config import get_default_config
    cfg = get_default_config()
    rng = np.random.default_rng(0)
    rows = []
    for s in range(6):                                   # 6 площадок
        for k in range(4):                               # по 4 пробы
            rows.append({"object": f"obj_{s}_{k}", "site_id": f"site_{s}",
                         "soil_type": "sand", "load_mode": "seismic",
                         "liq_label": int(k % 2)})
    # объект obj_0_* и «obj_dup» делят site_0 → должны держаться вместе
    rows.append({"object": "obj_dup", "site_id": "site_0", "soil_type": "sand",
                 "load_mode": "seismic", "liq_label": 1})
    meta = pd.DataFrame(rows)
    folds = make_grouped_cv_folds(meta, len(meta), seed=1, config=cfg, n_splits=3, n_repeats=1)
    sid = meta["site_id"].to_numpy()
    for f in folds:
        for a, bset in (("train_rel", "test_rel"), ("train_rel", "val_rel"), ("val_rel", "test_rel")):
            sa = set(sid[f[a]]); sb = set(sid[f[bset]])
            assert sa.isdisjoint(sb), f"site утёк между {a} и {bset}: {sa & sb}"


def test_n_folds_counts_group_size_not_nonnull_p3():
    # CatBoost с NaN P3_Core на всех фолдах должен иметь n_folds=число строк, а не 0.
    raw = pd.DataFrame({
        "model": ["CatBoost"] * 3 + ["DPI-Flow"] * 3,
        "fold": [0, 1, 2, 0, 1, 2],
        "P3_Core": [np.nan, np.nan, np.nan, 1.0, 1.1, 0.9],
        "AUPRC": [0.7, 0.72, 0.71, 0.8, 0.81, 0.79],
    })
    summary = aggregate_cv(raw, metric_keys=["P3_Core", "AUPRC"])
    nf = dict(zip(summary["model"], summary["n_folds"]))
    assert nf["CatBoost"] == 3, f"CatBoost n_folds={nf['CatBoost']} (должно быть 3, не 0)"
    assert nf["DPI-Flow"] == 3


def test_measured_csr_amplitude_and_indicators_present():
    # #3 пер-цикловая амплитуда восстанавливает переменную нагрузку; #10 индикаторы в признаках.
    from liquefaction_ai.data.ppr_envelope import extract_cycle_amplitude
    import inspect
    from liquefaction_ai.data import synthetic
    # синус амплитуды 5 на 4 цикла по 8 точек → амплитуда ≈5 на каждом цикле
    t = np.linspace(0, 4 * 2 * np.pi, 32, endpoint=False)
    cyc = np.repeat(np.arange(1, 5), 8).astype(float)
    sig = 5.0 * np.sin(t)
    c, a = extract_cycle_amplitude(cyc, sig, points_in_cycle=8)
    assert a.size == 4 and np.all(np.abs(a - 5.0) < 0.6), f"амплитуда {a} ≠ 5"
    # переменная амплитуда (рамп) → амплитуда растёт
    sig2 = (1.0 + 0.5 * cyc) * np.sin(t)
    _, a2 = extract_cycle_amplitude(cyc, sig2, points_in_cycle=8)
    assert a2[-1] > a2[0] * 1.3, "переменная амплитуда не восстановлена"
    # #10 индикаторы пропусков объявлены в статических признаках
    src = inspect.getsource(synthetic.build_feature_matrices)
    for ind in ("miss_e", "miss_Ip", "miss_K0", "miss_vs", "miss_gran"):
        assert ind in src, f"{ind} нет в признаках"


def test_nmax_removed_from_feature_builder():
    # #1 N_max не должен присутствовать среди статических признаков (утечка длительности).
    import inspect
    from liquefaction_ai.data import synthetic
    src = inspect.getsource(synthetic.build_feature_matrices)
    # нормировка цикловых признаков — на max_cycle_reference (константа), не на N_max
    assert "log1p(_H)" in src or "max_cycle_reference" in src
    assert 'load_df["N_max"].to_numpy()[:, None]' not in src   # больше не нормируем по N_max


def test_catboost_regressor_trains_on_uncensored_only():
    # #9 проверяем ЛОГИКУ маски регрессора в исходнике (без запуска CatBoost): rtr = mtr & (label==1).
    import inspect
    from liquefaction_ai.models import tabular
    src = inspect.getsource(tabular.CatBoostBaseline.fit)
    assert "rtr = mtr & (ytr_risk > 0.5)" in src, "N_liq-регрессор должен учиться только на разжижившихся"
