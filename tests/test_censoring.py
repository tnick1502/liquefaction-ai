"""
Тесты цензур-протокола N_liq (три режима опыта).

Проверяют:
* функцию потерь :func:`masked_censored_nliq_loss` — полную ошибку для разжижения, одностороннюю
  (Tobit) цензуру для стабилизации и исключение по маске для 3-го режима;
* классификатор режимов :func:`_terminal_observability` — разжижение / стабилизация / рост-без-исхода.
"""
import numpy as np
import torch

from liquefaction_ai.training.losses import masked_censored_nliq_loss
from liquefaction_ai.data.splits import _terminal_observability


def test_liquefied_is_full_smooth_l1():
    pred = torch.tensor([0.5]); tgt = torch.tensor([0.5]); lab = torch.tensor([1.0])
    assert masked_censored_nliq_loss(pred, tgt, lab).item() == 0.0
    # занижение разжижения штрафуется
    assert masked_censored_nliq_loss(torch.tensor([0.2]), tgt, lab).item() > 0.0


def test_stabilized_is_one_sided_tobit():
    tgt = torch.tensor([0.5]); lab = torch.tensor([0.0])
    # «перелёт» (pred > N_max-таргет) для нецензурированного НЕ штрафуется
    assert masked_censored_nliq_loss(torch.tensor([0.9]), tgt, lab).item() == 0.0
    # занижение (предсказали разжижение раньше точки цензуры) штрафуется
    assert masked_censored_nliq_loss(torch.tensor([0.1]), tgt, lab).item() > 0.0


def test_observed_mask_excludes_third_regime():
    pred = torch.tensor([0.2, 0.2]); tgt = torch.tensor([0.9, 0.9]); lab = torch.tensor([1.0, 1.0])
    # без маски учитываются оба; с маской — только первый образец (второй — 3-й режим)
    masked = masked_censored_nliq_loss(pred, tgt, lab, torch.tensor([1.0, 0.0])).item()
    only_first = masked_censored_nliq_loss(pred[:1], tgt[:1], lab[:1], None).item()
    assert abs(masked - only_first) < 1e-6


def test_all_masked_out_is_finite_zero():
    pred = torch.tensor([0.2]); tgt = torch.tensor([0.9]); lab = torch.tensor([1.0])
    loss = masked_censored_nliq_loss(pred, tgt, lab, torch.tensor([0.0])).item()
    assert np.isfinite(loss) and loss == 0.0  # деление на clamp(min=1), не на 0


def test_terminal_observability_three_regimes():
    seq = 20
    r = np.zeros((3, seq), dtype=np.float32)
    vm = np.ones((3, seq), dtype=np.float32)
    lab = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    r[0] = np.linspace(0.0, 1.0, seq)    # режим 2: разжижение
    r[1] = 0.5                           # режим 1: стабилизация (плоский хвост)
    r[2] = np.linspace(0.0, 0.85, seq)   # режим 3: растёт, не дошла
    obs = _terminal_observability(r, vm, lab)
    assert obs[0] == 1.0   # разжижение — терминал наблюдаем
    assert obs[1] == 1.0   # стабилизация — корректная право-цензура
    assert obs[2] == 0.0   # 3-й режим — терминал неоценим → исключаем
