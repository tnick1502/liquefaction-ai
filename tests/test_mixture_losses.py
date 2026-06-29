"""
Тесты proper-scoring лоссов предиктивной смеси (#3): gaussian_mixture_nll и energy_crps.

Эти лоссы привязывают разброс flow-постериора к предиктивной ошибке — без них поток не
калибруется и не даёт выигрыша по NLL/CRPS/покрытию.
"""
import math

import torch

from liquefaction_ai.training.losses import gaussian_nll, gaussian_mixture_nll, energy_crps


def test_mixture_of_identical_components_equals_single_nll():
    torch.manual_seed(0)
    B, T = 4, 6
    mean = torch.randn(B, T)
    logvar = torch.zeros(B, T)
    target = torch.randn(B, T)
    mask = torch.ones(B, T)
    # gaussian_nll опускает константу 0.5·log(2π); микстура — proper density (с константой).
    single = gaussian_nll(mean, logvar, target, mask) + 0.5 * math.log(2.0 * math.pi)
    # смесь из 5 ИДЕНТИЧНЫХ компонент = та же плотность
    mix = gaussian_mixture_nll(mean.expand(5, B, T), logvar.expand(5, B, T), target, mask)
    assert torch.allclose(single, mix, atol=1e-5), (single.item(), mix.item())
    # и инвариантность к числу одинаковых компонент
    mix1 = gaussian_mixture_nll(mean.unsqueeze(0), logvar.unsqueeze(0), target, mask)
    assert torch.allclose(mix1, mix, atol=1e-5)


def test_energy_crps_deterministic_equals_mae():
    B, T = 3, 5
    c = torch.full((B, T), 0.4)
    target = torch.full((B, T), 0.1)
    mask = torch.ones(B, T)
    # все сэмплы одинаковы → второй член 0 → CRPS = |c − y|
    crps = energy_crps(c.expand(8, B, T), target, mask)
    assert torch.allclose(crps, torch.tensor(0.3), atol=1e-6)


def test_well_spread_mixture_beats_misplaced_narrow_gaussian():
    B, T = 16, 8
    target = torch.zeros(B, T)
    mask = torch.ones(B, T)
    # смесь, покрывающая цель (компоненты вокруг 0, умеренная σ)
    means_good = torch.linspace(-1.0, 1.0, 6).view(6, 1, 1).expand(6, B, T)
    logv_good = torch.full((6, B, T), math.log(0.5 ** 2))
    nll_good = gaussian_mixture_nll(means_good, logv_good, target, mask)
    # узкая компонента, смещённая от цели
    nll_bad = gaussian_mixture_nll(torch.full((1, B, T), 1.5), torch.full((1, B, T), math.log(0.1 ** 2)),
                                   target, mask)
    assert nll_good < nll_bad


def test_mixture_nll_is_differentiable():
    means = torch.randn(4, 3, 5, requires_grad=True)
    logvars = torch.zeros(4, 3, 5, requires_grad=True)
    loss = gaussian_mixture_nll(means, logvars, torch.randn(3, 5), torch.ones(3, 5))
    loss.backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()
