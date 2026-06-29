"""
Тест интеграции MC-микстуры в DPIFlow (#3): opt-in, обратная совместимость, дифференцируемость.
"""
import torch

from liquefaction_ai.models.dpi_flow import DPIFlow

_KW = dict(static_dim=20, prefix_dim=6, seq_len=24, prefix_len=6, max_cycle_reference=3000.0)


def test_mc_flags_default_off_and_stored():
    m = DPIFlow(**_KW)
    assert m.mc_train_samples == 0 and m.mc_crps_weight == 0.0   # по умолчанию выкл → поведение прежнее
    m2 = DPIFlow(**_KW, mc_train_samples=4, mc_crps_weight=0.5, mc_predict_samples=16)
    assert m2.mc_train_samples == 4 and m2.mc_crps_weight == 0.5 and m2.mc_predict_samples == 16


def test_weights_backward_compatible():
    # модель с выключенным MC должна грузить state_dict обычной модели (новые флаги — не параметры)
    a = DPIFlow(**_KW)
    b = DPIFlow(**_KW, mc_train_samples=8, mc_crps_weight=0.3)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True), None
    assert missing.missing_keys == [] and missing.unexpected_keys == []


def test_dpi_evt_mc_flags_and_compat():
    # #3 зеркально в DPI-EVT: opt-in флаги, default off, совместимость весов, наличие predictive()
    from liquefaction_ai.models.dpi_evt import DPIEvtNet
    kw = dict(static_dim=20, prefix_dim=6, seq_dim=5, seq_len=24, prefix_len=6, max_cycle_reference=3000.0)
    a = DPIEvtNet(**kw)
    assert a.mc_train_samples == 0 and a._force_sample is False and hasattr(a, "predictive")
    b = DPIEvtNet(**kw, mc_train_samples=4, mc_crps_weight=0.3)
    r = b.load_state_dict(a.state_dict(), strict=True)
    assert r.missing_keys == [] and r.unexpected_keys == []
