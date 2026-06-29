"""
Тест leakage-free fixed-prefix протокола (prefix_mode='fixed_k').

В основном протоколе ('preonset') длина префикса зависит от момента onset → коррелирует с
исходом (канал утечки). В fixed_k префикс = первые K шагов для ВСЕХ опытов, длина не зависит
от исхода.
"""
import numpy as np

from liquefaction_ai.config import get_default_config
from liquefaction_ai.data.real_adapter import build_observed_prefix


def _toy():
    # 2 быстрых (onset рано) + 2 медленных опыта, одинаковая валидная длина
    seq = 20
    r = np.zeros((4, seq), dtype=np.float32)
    r[0, 3:] = 0.99      # onset на шаге 3 (быстрый, разжижается)
    r[1, 4:] = 0.99      # onset на шаге 4
    r[2] = np.linspace(0, 0.5, seq)   # не доходит
    r[3] = np.linspace(0, 0.4, seq)
    vm = np.ones((4, seq), dtype=np.float32)
    return r, vm


def test_fixed_k_prefix_is_outcome_independent():
    r, vm = _toy()
    K = 6
    obs = build_observed_prefix(r, vm, K, strict_preonset=False)   # режим fixed_k
    lengths = obs["prefix_mask"].sum(axis=1)
    assert np.allclose(lengths, K), f"fixed_k должен давать ровно K={K} шагов, получено {lengths}"
    # ни одной post-window точки во входе
    assert obs["prefix_mask"][:, K:].sum() == 0


def test_preonset_prefix_length_depends_on_onset():
    r, vm = _toy()
    obs = build_observed_prefix(r, vm, prefix_len=12, strict_preonset=True, margin=1)
    lengths = obs["prefix_mask"].sum(axis=1)
    # у быстрых опытов (ранний onset) префикс короче, чем у медленных → длина зависит от исхода
    assert lengths[0] < lengths[2], "preonset: ранний onset должен укорачивать префикс"


def test_config_exposes_fixed_prefix_protocol():
    cfg = get_default_config()
    assert cfg.prefix_mode == "preonset"          # основной протокол по умолчанию
    assert isinstance(cfg.prefix_fixed_k, int) and cfg.prefix_fixed_k > 0
