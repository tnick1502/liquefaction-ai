"""
Тесты двух P0-исправлений протокола:

  * #1/#3 — фактическая длительность опыта НЕ попадает в endpoint входной сетки (он = ПЛАНОВЫЙ
    горизонт, a-priori), а N_liq определяется на поцикловой огибающей (поцикловое разрешение);
  * #6 — censor time неразжижившегося = ФАКТический last_obs, а не плановый горизонт;
  * #2 — empirical object-level coverage не является ранговой тавтологией;
  * #7 — synthetic-сетка строится по ЯВНЫМ аргументам (без global state, order-independent).
"""
from types import SimpleNamespace

import numpy as np

from liquefaction_ai.config import LIQ_THRESHOLD
from liquefaction_ai.data.raw_loader import extract_test
from liquefaction_ai.data.synthetic import build_log_dense_cycles
from liquefaction_ai.evaluation.metrics import (conformal_band_quantile,
                                                per_trajectory_nonconformity)
from liquefaction_ai.evaluation.cross_validation import aggregate_object_conformal


def _make_objs(ppr_curve, cyc, planned_cycles=3000.0):
    """Сконструировать минимальные data/handler-объекты под extract_test (gv-доступ к __dict__)."""
    data = SimpleNamespace(cycles=cyc, PPR=ppr_curve, deviator=ppr_curve * 50.0, strain=ppr_curve * 0.1)
    phys = SimpleNamespace(type_ground=7, e=0.7)
    tp = SimpleNamespace(physical=phys, points_in_cycle=None, sigma_1=100.0, t=20.0,
                         cycles_count=planned_cycles, frequency=0.5, K0=0.5)
    handler = SimpleNamespace(_test_params=tp, _test_result=None)
    return data, handler


def test_grid_endpoint_is_planned_horizon_not_last_obs():
    # Опыт разжижается и обрывается на onset (last_obs≈150), но плановый горизонт = 3000.
    cyc = np.arange(1, 151, dtype=float)
    ppr = np.clip(np.linspace(0.0, 1.05, 150), 0, 1.05)        # пересекает 0.95 внутри окна
    data, handler = _make_objs(ppr, cyc, planned_cycles=3000.0)
    soil, load, arr, liq, n_liq = extract_test(data, handler, "Потенциал разжижения", 72,
                                               landmark_n0=20.0, landmark_k=12, horizon_default=3000.0)
    grid = arr["cycles"]; mask = arr["mask"]
    # endpoint входной сетки = ПЛАНОВЫЙ горизонт (a-priori), НЕ фактический last_obs≈150
    assert grid[-1] > 1500.0, f"endpoint сетки={grid[-1]:.0f} утёк к last_obs вместо планового 3000"
    assert abs(grid[-1] - 3000.0) / 3000.0 < 0.05
    # mask валиден ТОЛЬКО до последнего наблюдённого цикла (доступность таргета, не длина сетки)
    last_valid = grid[mask > 0].max()
    assert last_valid <= 155.0, f"маска валидна до {last_valid:.0f} — длительность опыта утекает в таргет"
    assert liq == 1


def test_landmark_prefix_is_invariant_to_post_landmark_values():
    """Изменение PPR после N0 не должно менять ни одной входной prefix-точки."""
    cyc = np.arange(1, 3001, dtype=float)
    ppr_a = np.zeros_like(cyc); ppr_a[20:] = 1.0
    ppr_b = np.zeros_like(cyc); ppr_b[1000:] = 0.7
    da, ha = _make_objs(ppr_a, cyc); db, hb = _make_objs(ppr_b, cyc)
    aa = extract_test(da, ha, "Потенциал разжижения", 72, 20.0, 12, 3000.0)[2]
    ab = extract_test(db, hb, "Потенциал разжижения", 72, 20.0, 12, 3000.0)[2]
    assert np.allclose(aa["prefix_r"][:12], ab["prefix_r"][:12])
    assert np.allclose(aa["r_causal"][:12], ab["r_causal"][:12])
    assert not np.allclose(aa["r"][:12], ab["r"][:12])  # target вправе использовать полную кривую


def test_nliq_per_cycle_resolution_and_event():
    # N_liq берётся на поцикловой огибающей → не квантуется редкими поздними узлами сетки.
    cyc = np.arange(1, 201, dtype=float)
    ppr = np.clip(np.linspace(0.0, 1.10, 200), 0, 1.10)
    cross_true = int(np.argmax(ppr >= LIQ_THRESHOLD)) + 1     # ~цикл первого пересечения
    data, handler = _make_objs(ppr, cyc, planned_cycles=3000.0)
    _, _, _, liq, n_liq = extract_test(data, handler, "Потенциал разжижения", 72,
                                       landmark_n0=20.0, landmark_k=12, horizon_default=3000.0)
    assert liq == 1
    # поцикловое разрешение: |N_liq − истинный цикл пересечения| мал (а не «округлён» до узла сетки)
    assert abs(n_liq - cross_true) <= 5, f"N_liq={n_liq} далёк от пересечения {cross_true} (квантование сеткой)"


def test_nonliq_censor_is_last_obs_not_planned():
    # Неразжижившийся: PPR выходит на плато 0.5; censor time = ФАКТический last_obs (=800), не план 3000.
    cyc = np.arange(1, 801, dtype=float)
    ppr = np.clip(0.5 * (1 - np.exp(-cyc / 50.0)), 0, 1.0)     # плато ~0.5, порог не пересекает
    data, handler = _make_objs(ppr, cyc, planned_cycles=3000.0)
    _, _, arr, liq, n_liq = extract_test(data, handler, "Потенциал разжижения", 72,
                                         landmark_n0=20.0, landmark_k=12, horizon_default=3000.0)
    assert liq == 0
    assert abs(n_liq - 800.0) <= 5.0, f"censor N_liq={n_liq} должен быть last_obs≈800, не плановый 3000"
    # endpoint входной сетки при этом всё равно = плановый горизонт (a-priori)
    assert arr["cycles"][-1] > 1500.0


def test_band_quantile_is_not_a_rank_tautology():
    # conformal_band_quantile должен ЗАВИСЕТЬ от масштаба скоров (в отличие от прежней ранговой
    # «LOO-coverage», которая возвращала ~0.9 для любых значений). Это калибровочный квантиль.
    q_small = conformal_band_quantile(np.linspace(0.1, 1.0, 50), level=0.90)
    q_big = conformal_band_quantile(np.linspace(0.1, 1.0, 50) * 1e6, level=0.90)
    assert q_big > 1e5 * q_small * 0.5, "квантиль не масштабируется со скорами — снова тавтология"
    assert np.isnan(conformal_band_quantile([]))
    nc = per_trajectory_nonconformity(np.zeros((1, 4)), np.ones((1, 4)),
                                      np.array([[0, 0, 3.0, 0]]), np.ones((1, 4)))
    assert abs(float(nc[0]) - 3.0) < 1e-9


def test_empirical_site_held_out_coverage_measures_real_coverage():
    # Покрытие меряется на TEST-объектах при q, калиброванном вне их. Конструируем случай, где
    # половина объектов превышает q → покрытие ≈0.5 (а НЕ авто-0.9). Это доказывает не-тавтологичность.
    import pandas as pd
    rows = []
    for o in range(20):
        s = 0.5 if o % 2 == 0 else 5.0          # чётные покрыты (≤q=1.0), нечётные — нет
        rows.append({"model": "A", "repeat": 0, "fold": o % 5, "object": f"o{o}",
                     "nonconf_max": s, "conf_q_val": 1.0, "conf_band_width": 2.0})
    res = aggregate_object_conformal(pd.DataFrame(rows), level=0.90, n_boot=500)
    cov = float(res.loc[res.model == "A", "Coverage_emp"].iloc[0])
    assert abs(cov - 0.5) < 1e-6, f"empirical coverage {cov} не отражает реальную долю покрытых объектов"
    assert float(res.loc[0, "mean_band_width"]) == 2.0
    assert int(res.loc[0, "n_objects"]) == 20


def test_synthetic_grid_explicit_args_order_independent():
    nmax = np.array([50.0, 500.0, 3000.0])
    a, _ = build_log_dense_cycles(nmax, 72, landmark_n0=20.0, landmark_k=12)
    b, _ = build_log_dense_cycles(nmax, 72)                       # обычная сетка
    # повтор в другом порядке — результат идентичен (нет global state)
    b2, _ = build_log_dense_cycles(nmax, 72)
    a2, _ = build_log_dense_cycles(nmax, 72, landmark_n0=20.0, landmark_k=12)
    assert np.allclose(a, a2) and np.allclose(b, b2)
    assert not np.allclose(a, b)
    assert np.allclose(a[:, :12], a[0, :12])                      # общий ранний grid
