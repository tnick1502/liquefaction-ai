import numpy as np
import pandas as pd
import pytest
import torch

from liquefaction_ai.config import get_default_config
from liquefaction_ai.evaluation.ablation_study import paired_ablation_summary
from liquefaction_ai.evaluation.cross_validation import (
    nliq_tail_tables,
    publication_model_kwargs,
)
from liquefaction_ai.evaluation.manifest import publication_preflight
from liquefaction_ai.evaluation.metrics import compute_metrics
from liquefaction_ai.evaluation.p3_ranking import compute_physical_admissibility
from liquefaction_ai.training.losses import (
    monotone_residual_scale,
    normalized_free_increments,
)


def _metric_split():
    return {
        "meta": pd.DataFrame({"object": ["a", "b"], "site_id": ["s1", "s2"],
                              "soil_type": ["sand", "silt"]}),
        "label": torch.tensor([0.0, 0.0]),
        "n_liq_true": torch.tensor([3000.0, 3000.0]),
        "nliq_censor_valid": torch.ones(2),
        "risk_label_observed": torch.ones(2),
        "r_obs": torch.zeros(2, 4),
        # Only the first two points were observed. PVR must still audit the full predicted horizon.
        "mask": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        "prefix_mask": torch.zeros(2, 4),
    }


def test_pvr_enforces_unit_bounds_and_monotonicity_on_full_forecast_grid():
    outputs = {
        "risk_prob": np.array([0.1, 0.1]),
        "nliq": np.array([3000.0, 3000.0]),
        # Violations occur only outside the observed mask: decrease in row 0, ru>1 in row 1.
        "traj_mean": np.array([[0.1, 0.2, 0.19, 0.3], [0.1, 0.2, 0.9, 1.01]]),
    }
    metrics, samples = compute_metrics("pvr", outputs, _metric_split(), get_default_config())
    assert metrics["Physics_Violation_Rate"] == 1.0
    assert samples["physics_decrease_violation"].tolist() == [1.0, 0.0]
    assert samples["physics_bounds_violation"].tolist() == [0.0, 1.0]
    assert compute_physical_admissibility(0.0) == (False, 0.0, 1.0)
    assert compute_physical_admissibility(1e-6)[0] is True
    assert compute_physical_admissibility(1e-6)[2] == 0.0


def test_free_increment_total_is_grid_independent_and_near_noop_at_init():
    gate = torch.tensor(-6.0)
    totals = []
    for steps in (8, 72):
        inc = normalized_free_increments(torch.zeros(3, steps), gate)
        totals.append(inc.sum(dim=1))
        corrected = monotone_residual_scale(torch.zeros(3, steps), torch.zeros(3, steps),
                                            free_increment=inc)
        assert torch.allclose(corrected[:, -1], inc.sum(dim=1), atol=1e-7)
    assert torch.allclose(totals[0], totals[1], atol=1e-7)
    assert float(totals[0].max()) < 0.003
    with pytest.raises(ValueError, match="span"):
        monotone_residual_scale(torch.zeros(1, 3), torch.zeros(1, 3), span=1.0)


def test_publication_model_contract_overrides_stale_artifact_switches():
    cfg = get_default_config()
    flow = publication_model_kwargs("dpi_flow", {
        "static_dim": 3, "max_cycle_reference": 1500.0,
        "probabilistic": False, "use_free_increment": True,
    }, cfg)
    assert flow["probabilistic"] is True
    assert flow["use_flow"] is True
    assert flow["calibration_steps"] == 0
    assert flow["use_free_increment"] is False
    assert flow["max_cycle_reference"] == cfg.max_cycle_reference

    evt = publication_model_kwargs("dpi_evt", {"probabilistic": False}, cfg)
    assert evt["probabilistic"] is True
    assert evt["nliq_from_curve"] is True
    assert evt["report_nliq_from_curve"] is True


def test_nliq_tail_audit_uses_one_oof_repeat_and_preserves_coherence():
    rows = []
    for repeat in (0, 1):
        for model, errors in (("DPI-Flow", (10.0, 1200.0)), ("PINN", (20.0, 30.0))):
            for i, err in enumerate(errors):
                rows.append({
                    "repeat": repeat, "fold": i, "model": model, "site_id": f"s{i}",
                    "object": f"o{i}", "soil_type": "sand", "liq_label": float(i == 0),
                    "N_liq_true": 3000.0, "nliq_pred": 3000.0 - err,
                    "nliq_abs_err": err, "risk_curve_coherent": float(i == 0),
                })
    detail, summary = nliq_tail_tables(pd.DataFrame(rows), top_k_per_model=1)
    assert len(detail) == 2
    assert detail.loc[detail["model"] == "DPI-Flow", "nliq_abs_err"].iloc[0] == 1200.0
    flow = summary.set_index("model").loc["DPI-Flow"]
    assert flow["n_valid"] == 2
    assert flow["N_error_ge_1000"] == 1


def test_publication_preflight_rejects_smoke_or_incomplete_protocol(tmp_path):
    with pytest.raises(RuntimeError, match="QUICK"):
        publication_preflight(tmp_path, quick=True, nested=True, run_loo=True,
                              run_ablations=True, output_root=tmp_path / "results",
                              require_clean=False)
    with pytest.raises(RuntimeError, match="RUN_LOO"):
        publication_preflight(tmp_path, quick=False, nested=True, run_loo=False,
                              run_ablations=True, output_root=tmp_path / "results",
                              require_clean=False)


def test_ablation_equivalence_averages_seeds_within_fold():
    rows = []
    for fold in range(3):
        for seed in (1, 2):
            rows.append({"tag": "g", "fold": fold, "seed": seed, "ablation": "full",
                         "Traj_RMSE_continuation": 0.100})
            rows.append({"tag": "g", "fold": fold, "seed": seed, "ablation": "variant",
                         "Traj_RMSE_continuation": 0.102})
    out = paired_ablation_summary(pd.DataFrame(rows),
                                  margins={"Traj_RMSE_continuation": 0.005}, n_boot=100)
    row = out.iloc[0]
    assert row["n_folds"] == 3 and row["n_seeds"] == 2
    assert abs(row["delta_worse_mean"] - 0.002) < 1e-9
    assert bool(row["practically_equivalent"])
