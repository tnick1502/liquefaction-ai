import torch

from liquefaction_ai.models import DPIFlow


def test_dpi_flow_inference_does_not_read_full_curve_auxiliary_targets():
    model = DPIFlow(static_dim=3, prefix_dim=2, seq_len=6, prefix_len=2,
                    max_cycle_reference=100.0, hidden_dim=16, theta_dim=31,
                    calibration_steps=0).eval()
    cycles = torch.tensor([[1.0, 2.0, 5.0, 10.0, 30.0, 100.0]])
    batch = {
        "static": torch.randn(1, 3), "prefix_summary": torch.randn(1, 2),
        "prefix_obs": torch.tensor([[0.1, 0.2, 0, 0, 0, 0.0]]),
        "prefix_mask": torch.tensor([[1.0, 1.0, 0, 0, 0, 0.0]]),
        "cycles": cycles, "delta_cycles": torch.diff(cycles, prepend=torch.zeros(1, 1)),
        "csr": torch.full((1, 6), 0.2),
        "g_obs": torch.zeros(1, 6), "risk_proxy": torch.zeros(1),
    }
    with torch.no_grad():
        a = model.forward_batch(batch)
        changed = dict(batch)
        changed["g_obs"] = torch.ones(1, 6) * 999
        changed["risk_proxy"] = torch.ones(1) * 999
        b = model.forward_batch(changed)
    for key in ("traj_mean", "risk_prob", "nliq"):
        assert torch.equal(a[key], b[key]), f"test-time output {key} depends on full-curve auxiliary target"
