"""
Физические инварианты состояний EVT-движка (EVT-NeuralSSM / DPI-EVT / DPI-Flow).

Ловит регресс артефакта «EVT-NeuralSSM: триггер спадает на хвосте» и «PPR>1»:
  * событие разжижения ПОГЛОЩАЮЩЕЕ → триггер g(N) неубывающий (иначе образец «разразжижается»);
  * урон z(N) необратим → неубывающий;
  * поровое давление ru = PPR ограничено сверху единицей (ru≤1 физически, не 1.05);
  * траектория PPR(N) монотонно неубывающая (недренированное накопление).
"""
import torch

from liquefaction_ai.models import DPIFlow, DPIEvtNet
from liquefaction_ai.models.evt_ssm import EVTNeuralSSM


def _batch(B=6, T=12, P=2, SD=5):
    cyc = torch.logspace(0, 3, T).repeat(B, 1)                    # лог-сетка 1..1000
    return dict(
        static=torch.randn(B, 3), prefix_summary=torch.randn(B, P),
        prefix_obs=torch.rand(B, T) * 0.3,
        prefix_mask=torch.cat([torch.ones(B, 3), torch.zeros(B, T - 3)], 1),
        cycles=cyc, delta_cycles=torch.diff(cyc, prepend=cyc[:, :1] * 0),
        csr=torch.full((B, T), 0.3), seq_in=torch.randn(B, T, SD),
        g_obs=torch.zeros(B, T), risk_proxy=torch.zeros(B),
        r_obs=torch.rand(B, T).cummax(1).values, mask=torch.ones(B, T),
        label=torch.tensor([1., 0., 1., 0., 1., 0.]), n_liq_norm=torch.rand(B),
        n_liq_true=torch.rand(B) * 100 + 10,
        reached_horizon=torch.tensor([0., 1., 0., 1., 0., 1.]),
        regime_stable=torch.tensor([0., 1., 0., 1., 0., 1.]),
    )


def _models(T=12, P=2, SD=5):
    kw = dict(static_dim=3, prefix_dim=P, seq_len=T, prefix_len=3,
              max_cycle_reference=3000.0, hidden_dim=16)
    return [
        ("EVT-NeuralSSM", EVTNeuralSSM(seq_dim=SD, **kw)),
        ("DPI-EVT", DPIEvtNet(seq_dim=SD, use_flow=False, **kw)),
        ("DPI-Flow", DPIFlow(theta_dim=31, **kw)),
    ]


def test_trigger_and_damage_are_nondecreasing_and_ppr_bounded():
    torch.manual_seed(0)
    batch = _batch()
    for name, m in _models():
        out = m.forward_batch(batch)
        g = out["g"]
        assert float((g[:, 1:] - g[:, :-1]).min()) >= -1e-5, f"{name}: триггер g убывает (не поглощающее событие)"
        r = out["traj_mean"]
        assert float((r[:, 1:] - r[:, :-1]).min()) >= -1e-5, f"{name}: PPR немонотонна"
        assert float(r.max()) <= 1.0 + 1e-5, f"{name}: PPR>1 (ru должно быть ≤1)"
        if out.get("z") is not None:
            z = out["z"]
            assert float((z[:, 1:] - z[:, :-1]).min()) >= -1e-5, f"{name}: урон z убывает (необратимость)"


def test_loss_is_finite_and_differentiable_under_constraints():
    torch.manual_seed(0)
    batch = _batch()
    for name, m in _models():
        loss = m.compute_loss(batch)["loss"]
        assert torch.isfinite(loss), f"{name}: loss не конечен"
        loss.backward()


def test_reported_nliq_is_curve_first_coherent():
    # Все три физ-модели репортят N_liq = интерполированное пересечение СОБСТВЕННОЙ mean-PPR кривой
    # (coherence-gap ≈ 0). Ловит регресс к head-based репортингу, некогерентному с траекторией.
    from liquefaction_ai.training.losses import interpolated_crossing
    torch.manual_seed(0)
    batch = _batch()
    for name, m in _models():
        m.eval()
        with torch.no_grad():
            out = m.forward_batch(batch)
            curve_nliq = interpolated_crossing(out["traj_mean"], batch["cycles"], m.liq_threshold)
            gap = (out["nliq"] - curve_nliq).abs().max().item()
        assert gap < 1e-3, f"{name}: reported N_liq не на своей кривой (gap={gap:.4f})"
