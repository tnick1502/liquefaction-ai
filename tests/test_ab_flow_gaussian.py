"""
Тест A/B-блока «flow vs gaussian» (#3, исправлено): корректный кластерный bootstrap по объектам
(с кратностью ресэмпла), один оценщик для точки и CI.
"""
import numpy as np
import pandas as pd

from liquefaction_ai.evaluation.ab_test import _object_cluster_ab


def _frame(value, n_obj=8, per=10):
    objs = np.repeat([f"o{i}" for i in range(n_obj)], per)
    return pd.DataFrame({"object": objs, "nll": np.full(n_obj * per, value)})


def test_cluster_bootstrap_detects_clear_winner():
    sf = _frame(0.0)   # flow: NLL=0 на всех объектах
    sg = _frame(1.0)   # gaussian: NLL=1
    r = _object_cluster_ab(sf, sg, "nll", nboot=500, rng=np.random.default_rng(0))
    assert r["flow"] == 0.0 and r["gaussian"] == 1.0
    assert r["diff_gauss_minus_flow"] == 1.0          # gauss−flow = 1 (flow лучше)
    assert r["P(flow_better)"] == 1.0
    assert r["ci95_low"] <= r["diff_gauss_minus_flow"] <= r["ci95_high"]   # точка ВНУТРИ CI


def test_point_estimate_inside_ci_for_noisy_tie():
    rng = np.random.default_rng(0)
    objs = np.repeat([f"o{i}" for i in range(10)], 8)
    sf = pd.DataFrame({"object": objs, "nll": rng.normal(0.5, 0.1, 80)})
    sg = pd.DataFrame({"object": objs, "nll": rng.normal(0.5, 0.1, 80)})
    r = _object_cluster_ab(sf, sg, "nll", nboot=800, rng=np.random.default_rng(1))
    # точка обязана лежать внутри интервала (один оценщик для точки и CI — нет «невозможных» строк)
    assert r["ci95_low"] <= r["diff_gauss_minus_flow"] <= r["ci95_high"]
