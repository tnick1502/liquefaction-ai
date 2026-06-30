"""
Тесты единого канонического порога события разжижения.

Метка риска, N_liq, наблюдаемый триггер g_obs (auxiliary supervision) и момент пересечения
во всех моделях должны описывать ОДНО событие — пересечение ru=PPR порога :data:`LIQ_THRESHOLD`.
Эти тесты ловят рассинхронизацию порога (исторически: генератор 0.985, триггер 0.9, модели 0.95).
"""
import inspect

import numpy as np
import pandas as pd

from liquefaction_ai.config import LIQ_THRESHOLD, get_default_config
from liquefaction_ai.data.observed import derive_observed_targets


def test_canonical_threshold_is_single_source():
    assert LIQ_THRESHOLD == 0.95
    assert get_default_config().liq_threshold == LIQ_THRESHOLD


def test_censoring_horizons_are_explicit():
    cfg = get_default_config()
    assert cfg.max_cycle_reference == 3000.0
    assert cfg.min_nonliq_complete_cycles == 500.0


def test_observed_trigger_uses_canonical_threshold():
    default = inspect.signature(derive_observed_targets).parameters["liq_threshold"].default
    assert default == LIQ_THRESHOLD, "g_obs строится по другому порогу, чем событие"


def test_all_models_default_to_canonical_threshold():
    from liquefaction_ai.models.dpi_flow import AnalyticalLiquefactionLayer, DPIFlow
    from liquefaction_ai.models.dpi_evt import DPIEvtNet
    from liquefaction_ai.models.evt_ssm import EVTNeuralSSM

    for cls in (AnalyticalLiquefactionLayer, DPIFlow, DPIEvtNet, EVTNeuralSSM):
        default = inspect.signature(cls.__init__).parameters["liq_threshold"].default
        assert default == LIQ_THRESHOLD, f"{cls.__name__}: порог пересечения ≠ канону"

    hit_default = inspect.signature(AnalyticalLiquefactionLayer.soft_first_hitting).parameters["threshold"].default
    assert hit_default == LIQ_THRESHOLD
    assert "0.985" not in inspect.getsource(EVTNeuralSSM.soft_first_hitting)


def test_synthetic_event_definition_uses_canonical_threshold():
    from liquefaction_ai.data.synthetic import build_observations

    soil_df = pd.DataFrame({"D_r": [0.6, 0.6]})
    load_df = pd.DataFrame({"mode_id": [0, 0], "N_max": [99.0, 99.0]})
    cycles = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    r_true = np.array([[0.2, 0.6, 0.94, 0.97], [0.1, 0.3, 0.5, 0.9]], dtype=np.float32)
    # Высокий латентный триггер без пересечения PPR-порога не должен сам создавать событие.
    g_true = np.array([[0.0, 0.0, 0.0, 0.0], [0.99, 0.99, 0.99, 0.99]], dtype=np.float32)
    z_true = np.zeros_like(r_true)
    hidden = {"entropy": np.zeros(2, dtype=np.float32)}

    obs = build_observations(soil_df, load_df, hidden, z_true, r_true, g_true, cycles,
                             np.random.default_rng(123), prefix_len=1)

    assert obs["liq_label"].tolist() == [1.0, 0.0]
    # Для non-event правая цензура ставится на фактическом последнем наблюдённом цикле,
    # а не на плановом N_max, который может быть не достигнут.
    assert obs["n_liq_true"].tolist() == [4.0, 4.0]


def test_dpi_hitting_is_exact_forward_and_soft_backward():
    import torch
    from liquefaction_ai.models.dpi_flow import AnalyticalLiquefactionLayer

    layer = AnalyticalLiquefactionLayer(seq_len=2, max_cycle_reference=3000.0)
    cycles = torch.tensor([[100.0, 200.0]])

    crossing = torch.tensor([[0.94, 0.96]], requires_grad=True)
    n_cross = layer.soft_first_hitting(crossing, torch.zeros_like(crossing), cycles)
    assert torch.allclose(n_cross, torch.tensor([150.0]), atol=1e-3)

    below = torch.tensor([[0.90, 0.94]], requires_grad=True)
    n_below = layer.soft_first_hitting(below, torch.zeros_like(below), cycles)
    n_below.sum().backward()
    assert n_below.item() == 200.0                  # forward = right-censor at horizon
    assert torch.count_nonzero(below.grad).item() > 0  # backward всё ещё обучает onset
