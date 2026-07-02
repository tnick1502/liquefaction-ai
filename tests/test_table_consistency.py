"""
Тесты численной согласованности отчётных таблиц.

Ловят расхождения уровня «paper claim»: разные значения одной метрики в разных CSV, ненулевые
нарушения монотонности у физических моделей, отсутствие обязательных таблиц и согласованность
маски цензуры N_liq между обучением и метриками.
"""
import pandas as pd
import math
import numpy as np
import pytest
import torch

from conftest import TABLES
from liquefaction_ai.config import get_default_config
from liquefaction_ai.evaluation.metrics import compute_metrics, stress_split

ANALYSIS = TABLES.parent / "analysis_tables"
_HAS_PUBLICATION_RUN = ((TABLES / "publication_headline_grouped.csv").exists()
                        and (ANALYSIS / "cv_grouped_raw.csv").exists()
                        and (TABLES.parent / "run_manifest.json").exists())
_skip = pytest.mark.skipif(not _HAS_PUBLICATION_RUN,
                           reason="нет полного publication run: grouped headline/raw + run_manifest")
PHYS = ["DPI-Flow", "DPI-EVT", "EVT-NeuralSSM"]


@_skip
def test_required_tables_exist():
    for name in ["publication_headline_grouped.csv", "publication_headline_loo.csv",
                 "p3_core_ranking.csv", "nliq_tail_cases.csv"]:
        assert (TABLES / name).exists(), f"нет таблицы {name}"
    for name in ["cv_grouped_raw.csv", "cv_grouped_samples.csv", "cv_grouped_summary.csv",
                 "cv_loo_raw.csv", "cv_loo_samples.csv", "cv_loo_summary.csv",
                 "nliq_tail_summary.csv", "ablations.csv", "ablations_summary.csv",
                 "ablations_paired_equivalence.csv",
                 "ood_no_prefix.csv", "ab_flow_vs_gaussian_pooled.csv"]:
        assert (ANALYSIS / name).exists(), f"нет таблицы {name}"


@_skip
def test_only_notebook_generated_tables_are_committed():
    allowed = {
        "publication_headline_grouped.csv",
        "publication_headline_loo.csv",
        "p3_core_ranking.csv",
        "nliq_tail_cases.csv",
    }
    present = {p.name for p in TABLES.glob("*.csv")}
    assert present <= allowed, f"лишние CSV вне notebook-пайплайна: {sorted(present - allowed)}"


@_skip
def test_p3_matches_primary_grouped_estimator():
    raw = pd.read_csv(ANALYSIS / "cv_grouped_raw.csv")
    raw = raw[raw["repeat"] == raw["repeat"].min()]
    lb = raw.groupby("model").mean(numeric_only=True)
    p3 = pd.read_csv(TABLES / "p3_core_ranking.csv").set_index("model")
    for m in PHYS:
        for col in ["N_liq_logMAE", "Traj_RMSE", "AUPRC", "Brier"]:
            assert abs(float(lb.loc[m, col]) - float(p3.loc[m, col])) < 1e-3, \
                f"{m}.{col}: grouped primary estimator и p3_core_ranking расходятся"


@_skip
def test_physics_models_zero_monotonicity_violation():
    lb = pd.read_csv(ANALYSIS / "cv_grouped_raw.csv").groupby("model").mean(numeric_only=True)
    for m in PHYS:
        assert float(lb.loc[m, "Physics_Violation_Rate"]) == 0.0


@_skip
def test_nliq_metric_is_censored():
    # Метрики N_liq считаются по тому же цензур-протоколу, что и обучение: разжижение
    # оценивается точной ошибкой, любой валидный non-liq — односторонней ошибкой от censor time.
    lb = pd.read_csv(ANALYSIS / "cv_grouped_raw.csv")
    assert "N_liq_n_observed" in lb.columns, "метрика N_liq не помечена цензур-маской"
    # Модели с цензур-совместимым N_liq должны иметь >0 наблюдаемых. Модели, чей N_liq — ДРУГОЙ
    # estimand (напр. CatBoost: регрессор обучен только на разжижившихся → censored-aware метрика N/A),
    # помечаются N_liq_n_observed=0 ОСОЗНАННО, чтобы не сравнивать разные величины. Их исключаем.
    _na_estimand = {"CatBoost"}
    _cmp = lb[~lb["model"].isin(_na_estimand)]
    assert (_cmp["N_liq_n_observed"] > 0).all(), "censored-aware модель с пустой N_liq-маской"

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


def test_empty_continuation_is_nan_not_perfect_zero():
    split = {
        "meta": pd.DataFrame({"soil_type": ["sand"]}),
        "label": torch.tensor([0.0]),
        "n_liq_true": torch.tensor([10.0]),
        "risk_label_observed": torch.tensor([0.0]),
        "nliq_censor_valid": torch.tensor([1.0]),
        "r_obs": torch.tensor([[0.1, 0.2]]),
        "mask": torch.tensor([[1.0, 1.0]]),
        "prefix_mask": torch.tensor([[1.0, 1.0]]),
    }
    outputs = {
        "risk_prob": np.array([0.2]), "nliq": np.array([20.0]),
        "traj_mean": np.array([[0.1, 0.2]]), "traj_logvar": np.zeros((1, 2)),
    }
    metrics, samples = compute_metrics("empty-cont", outputs, split, get_default_config())
    assert math.isnan(metrics["Traj_RMSE_continuation"])
    assert math.isnan(metrics["Coverage_90"])
    assert math.isnan(samples["traj_rmse_continuation"].iloc[0])
    assert math.isnan(samples["coverage90"].iloc[0])


def test_sample_nliq_truth_and_curve_coherence_are_unambiguous():
    threshold = get_default_config().liq_threshold
    crossing = 1.0 + (threshold - 0.10) / (1.00 - 0.10) * 100.0
    split = {
        "meta": pd.DataFrame({"soil_type": ["sand", "sand"], "N_liq_true": [5000.0, 101.0]}),
        "label": torch.tensor([1.0, 0.0]),
        "n_liq_true": torch.tensor([100.0, 101.0]),
        "risk_label_observed": torch.ones(2),
        "nliq_censor_valid": torch.ones(2),
        "cycles": torch.tensor([[1.0, 101.0], [1.0, 101.0]]),
        "r_obs": torch.tensor([[0.10, 1.00], [0.10, 0.80]]),
        "mask": torch.ones(2, 2),
        "prefix_mask": torch.zeros(2, 2),
    }
    outputs = {
        "risk_prob": np.array([0.9, 0.1]),
        "nliq": np.array([crossing, 101.0]),
        "traj_mean": np.array([[0.10, 1.00], [0.10, 0.80]]),
    }
    metrics, samples = compute_metrics("coherence", outputs, split, get_default_config())
    assert samples["N_liq_true"].tolist() == [100.0, 101.0]
    assert samples["N_liq_raw"].tolist() == [5000.0, 101.0]
    assert metrics["N_liq_Curve_Coherence_MAE"] < 1e-6
    assert metrics["Risk_Curve_Coherence_Rate"] == 1.0
    assert samples["risk_curve_coherent"].tolist() == [1.0, 1.0]


def test_exact_only_regressor_is_not_reported_as_censor_aware():
    split = {
        "meta": pd.DataFrame({"soil_type": ["sand", "sand"]}),
        "label": torch.tensor([1.0, 0.0]),
        "n_liq_true": torch.tensor([100.0, 500.0]),
        "risk_label_observed": torch.tensor([1.0, 0.0]),
        "nliq_censor_valid": torch.tensor([1.0, 1.0]),
    }
    outputs = {
        "risk_prob": np.array([0.9, 0.2]), "nliq": np.array([110.0, 200.0]),
        "supports_censored_nliq": np.zeros(2),
    }
    metrics, samples = compute_metrics("exact-only", outputs, split, get_default_config())
    assert math.isnan(metrics["N_liq_logMAE"])
    assert not math.isnan(metrics["N_liq_logMAE_liq"])
    assert samples["nliq_log_err"].isna().all()


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
    lb = pd.read_csv(ANALYSIS / "cv_grouped_raw.csv").set_index("model")
    for m in PHYS + ["PINN"]:
        assert m in lb.index
    p3 = pd.read_csv(TABLES / "p3_core_ranking.csv").set_index("model")
    assert abs(float(p3.loc["DPI-Flow", "P3_Core_Raw_Score"]) - 100.0) < 1e-6
