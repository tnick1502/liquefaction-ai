"""
Тесты численной согласованности отчётных таблиц.

Ловят расхождения уровня «paper claim»: разные значения одной метрики в разных CSV, ненулевые
нарушения монотонности у физических моделей, отсутствие обязательных таблиц и согласованность
маски цензуры N_liq между обучением и метриками.
"""
import pandas as pd
import math
import pytest
import torch

from conftest import TABLES
from liquefaction_ai.config import get_default_config
from liquefaction_ai.evaluation.metrics import compute_metrics, stress_split

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
    # Метрики N_liq считаются по тому же цензур-протоколу, что и обучение: разжижение
    # оценивается точной ошибкой, стабилизированный non-liq — односторонней ошибкой,
    # а 3-й режим исключается.
    lb = pd.read_csv(TABLES / "full_leaderboard.csv")
    assert "N_liq_n_observed" in lb.columns, "метрика N_liq не помечена цензур-маской"
    assert (lb["N_liq_n_observed"] > 0).all()

    split = {
        "meta": pd.DataFrame({"soil_type": ["sand", "sand", "sand", "sand"]}),
        "label": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        "n_liq_true": torch.tensor([10.0, 100.0, 100.0, 30.0]),
        "n_liq_observed": torch.tensor([1.0, 1.0, 1.0, 0.0]),
    }
    outputs = {
        "risk_prob": pd.Series([0.9, 0.1, 0.1, 0.9]).to_numpy(),
        "nliq": pd.Series([12.0, 150.0, 90.0, 999.0]).to_numpy(),
    }
    metrics, samples = compute_metrics("mask-check", outputs, split, get_default_config())
    assert metrics["N_liq_n_observed"] == 3
    assert abs(metrics["N_liq_MAE"] - 4.0) < 1e-6  # (|12-10| + max(100-150,0) + max(100-90,0))/3
    assert samples["nliq_abs_err"].tolist()[:3] == [2.0, 0.0, 10.0]
    assert samples["n_liq_observed"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_all_unfinished_nliq_slice_reports_nan_not_fallback_error():
    split = {
        "meta": pd.DataFrame({"soil_type": ["sand", "sand"]}),
        "label": torch.tensor([0.0, 0.0]),
        "n_liq_true": torch.tensor([3000.0, 3000.0]),
        "n_liq_observed": torch.tensor([0.0, 0.0]),
    }
    outputs = {
        "risk_prob": pd.Series([0.1, 0.2]).to_numpy(),
        "nliq": pd.Series([10.0, 20.0]).to_numpy(),
    }
    metrics, samples = compute_metrics("all-unfinished", outputs, split, get_default_config())
    assert metrics["N_liq_n_observed"] == 0
    assert math.isnan(metrics["N_liq_MAE"])
    assert samples["nliq_abs_err"].isna().all()


def test_post_prefix_trajectory_metric_excludes_conditioning_prefix():
    split = {
        "meta": pd.DataFrame({"soil_type": ["sand"]}),
        "label": torch.tensor([1.0]),
        "n_liq_true": torch.tensor([30.0]),
        "n_liq_observed": torch.tensor([1.0]),
        "r_obs": torch.tensor([[0.0, 0.1, 0.9, 1.0]]),
        "mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "prefix_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    }
    outputs = {
        "risk_prob": pd.Series([0.9]).to_numpy(),
        "nliq": pd.Series([30.0]).to_numpy(),
        # Huge prefix error, perfect continuation. Full RMSE should see it; continuation should not.
        "traj_mean": pd.DataFrame([[1.0, 1.1, 0.9, 1.0]]).to_numpy(),
    }
    metrics, samples = compute_metrics("prefix-check", outputs, split, get_default_config())
    assert metrics["Traj_RMSE"] > 0.7
    assert abs(metrics["Traj_RMSE_continuation"]) < 1e-7
    assert abs(samples["traj_rmse_continuation"].iloc[0]) < 1e-7


def test_stress_split_can_remove_prefix_and_derived_auxiliary_targets():
    split = {
        "prefix_summary": torch.ones(2, 3),
        "prefix_summary_raw": torch.ones(2, 3),
        "prefix_obs": torch.ones(2, 4),
        "prefix_mask": torch.ones(2, 4),
        "seq_in": torch.ones(2, 4, 5),
        "seq_in_raw": torch.ones(2, 4, 5),
        "g_obs": torch.ones(2, 4),
        "risk_proxy": torch.ones(2),
        "label": torch.zeros(2),
    }
    stressed = stress_split(split, no_prefix=True, drop_derived_aux=True)
    for key in ("prefix_summary", "prefix_summary_raw", "prefix_obs", "prefix_mask"):
        assert torch.count_nonzero(stressed[key]) == 0
    assert torch.count_nonzero(stressed["seq_in"][..., -2:]) == 0
    assert "g_obs" not in stressed and "risk_proxy" not in stressed
    assert torch.count_nonzero(split["prefix_obs"]) > 0  # original is not mutated


@_skip
def test_leaderboard_has_three_proposed_and_pinn_reference():
    lb = pd.read_csv(TABLES / "full_leaderboard.csv").set_index("model")
    for m in PHYS + ["PINN"]:
        assert m in lb.index
    p3 = pd.read_csv(TABLES / "p3_core_ranking.csv").set_index("model")
    assert abs(float(p3.loc["PINN", "P3_Core_Raw_Score"]) - 100.0) < 1e-6  # PINN — опорная 100
