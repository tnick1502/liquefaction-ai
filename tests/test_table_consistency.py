"""
Тесты численной согласованности отчётных таблиц.

Ловят расхождения уровня «paper claim»: разные значения одной метрики в разных CSV, ненулевые
нарушения монотонности у физических моделей, отсутствие обязательных таблиц и согласованность
маски цензуры N_liq между обучением и метриками.
"""
import pandas as pd
import pytest
import torch

from conftest import TABLES
from liquefaction_ai.config import get_default_config
from liquefaction_ai.evaluation.metrics import compute_metrics

_skip = pytest.mark.skipif(not (TABLES / "full_leaderboard.csv").exists(),
                           reason="нет results/tables/full_leaderboard.csv")
PHYS = ["DPI-Flow", "DPI-EVT", "EVT-NeuralSSM"]


@_skip
def test_required_tables_exist():
    for name in ["full_leaderboard.csv", "leaderboard_risk.csv", "leaderboard_trajectory.csv",
                 "main_comparison.csv", "p3_core_ranking.csv",
                 "probabilistic_quality.csv", "ood_by_soil.csv", "ood_by_csr.csv"]:
        assert (TABLES / name).exists(), f"нет таблицы {name}"


@_skip
def test_only_notebook_generated_tables_are_committed():
    allowed = {
        "full_leaderboard.csv",
        "leaderboard_risk.csv",
        "leaderboard_trajectory.csv",
        "main_comparison.csv",
        "p3_core_ranking.csv",
        "probabilistic_quality.csv",
        "ood_by_soil.csv",
        "ood_by_csr.csv",
    }
    present = {p.name for p in TABLES.glob("*.csv")}
    assert present <= allowed, f"лишние CSV вне notebook-пайплайна: {sorted(present - allowed)}"


@_skip
def test_p3_matches_full_leaderboard():
    lb = pd.read_csv(TABLES / "full_leaderboard.csv").set_index("model")
    p3 = pd.read_csv(TABLES / "p3_core_ranking.csv").set_index("model")
    for m in PHYS:
        for col in ["N_liq_logMAE", "Traj_RMSE", "AUROC", "Brier"]:
            assert abs(float(lb.loc[m, col]) - float(p3.loc[m, col])) < 1e-3, \
                f"{m}.{col}: full_leaderboard и p3_core_ranking расходятся"


@_skip
def test_physics_models_zero_monotonicity_violation():
    lb = pd.read_csv(TABLES / "full_leaderboard.csv").set_index("model")
    for m in PHYS:
        assert float(lb.loc[m, "Physics_Violation_Rate"]) == 0.0


@_skip
def test_nliq_metric_is_censored():
    # Метрики N_liq считаются по тому же цензур-протоколу, что и обучение: число учтённых
    # объектов не превышает размер теста (3-й режим исключён).
    lb = pd.read_csv(TABLES / "full_leaderboard.csv")
    assert "N_liq_n_observed" in lb.columns, "метрика N_liq не помечена цензур-маской"
    assert (lb["N_liq_n_observed"] > 0).all()

    split = {
        "meta": pd.DataFrame({"soil_type": ["sand", "sand", "sand"]}),
        "label": torch.tensor([1.0, 1.0, 1.0]),
        "n_liq_true": torch.tensor([10.0, 20.0, 30.0]),
        "n_liq_observed": torch.tensor([1.0, 0.0, 1.0]),
    }
    outputs = {
        "risk_prob": pd.Series([0.9, 0.9, 0.9]).to_numpy(),
        "nliq": pd.Series([10.0, 999.0, 40.0]).to_numpy(),
    }
    metrics, samples = compute_metrics("mask-check", outputs, split, get_default_config())
    assert metrics["N_liq_n_observed"] == 2
    assert abs(metrics["N_liq_MAE"] - 5.0) < 1e-6
    assert samples["n_liq_observed"].tolist() == [1.0, 0.0, 1.0]


@_skip
def test_leaderboard_has_three_proposed_and_pinn_reference():
    lb = pd.read_csv(TABLES / "full_leaderboard.csv").set_index("model")
    for m in PHYS + ["PINN"]:
        assert m in lb.index
    p3 = pd.read_csv(TABLES / "p3_core_ranking.csv").set_index("model")
    assert abs(float(p3.loc["PINN", "P3_Core_Raw_Score"]) - 100.0) < 1e-6  # PINN — опорная 100
