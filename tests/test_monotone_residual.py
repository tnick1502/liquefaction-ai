"""
#108: обучаемый траекторный residual DPI-Flow/DPI-EVT МОНОТОННО-СОХРАНЯЮЩИЙ по построению —
кривая PPR(N) остаётся неубывающей при ЛЮБОМ residual, даже без post-hoc cummax-проекции. Это
гарантирует, что proposed-модели не попадают в «physically unreliable» из-за residual.
"""
import torch

from liquefaction_ai.training.losses import monotone_residual_scale


def test_monotone_residual_preserves_monotonicity_for_any_residual():
    torch.manual_seed(0)
    base = torch.cumsum(torch.rand(16, 40).abs(), dim=1)        # неубывающая базовая кривая
    for _ in range(20):
        res = 10.0 * torch.randn(16, 40)                        # произвольный (даже экстремальный) residual
        out = monotone_residual_scale(base, res, span=0.10)
        diffs = out[:, 1:] - out[:, :-1]
        assert float(diffs.min()) >= -1e-6, "residual нарушил монотонность"
    # начальная точка сохраняется
    assert torch.allclose(out[:, 0], base[:, 0])


def test_residual_changes_rate_not_direction():
    # при нулевом residual коррекция = тождество (gate=1)
    base = torch.cumsum(torch.rand(4, 30).abs(), dim=1)
    out = monotone_residual_scale(base, torch.zeros(4, 30), span=0.10)
    assert torch.allclose(out, base, atol=1e-5)
    # положительный residual ускоряет рост (итоговый уровень выше), отрицательный — замедляет
    up = monotone_residual_scale(base, torch.full((4, 30), 5.0), span=0.10)
    down = monotone_residual_scale(base, torch.full((4, 30), -5.0), span=0.10)
    assert float(up[:, -1].mean()) >= float(base[:, -1].mean()) >= float(down[:, -1].mean())
