"""
Тест simultaneous (траекторного) conformal-покрытия: покрытие меряется по ВСЕЙ кривой
(band conformal), а не по независимым точкам.
"""
import numpy as np

from liquefaction_ai.evaluation.metrics import (simultaneous_conformal_coverage,
                                                split_conformal_coverage)


def _block(seed, N=400, T=30, sigma=0.1):
    r = np.random.default_rng(seed)
    pred = np.zeros((N, T)); std = np.full((N, T), sigma)
    true = pred + r.normal(0, sigma, (N, T))
    return pred, std, true, np.ones((N, T))


def test_simultaneous_hits_nominal_per_trajectory():
    cal = _block(1); test = _block(2)
    cov, w = simultaneous_conformal_coverage(*cal, *test, level=0.90)
    assert 0.85 <= cov <= 0.95 and w > 0          # ~90% ТРАЕКТОРИЙ целиком в полосе


def test_simultaneous_band_is_wider_than_pointwise():
    # simultaneous квантиль = max_t |y-ŷ|/σ по траектории ⇒ полоса ШИРЕ pointwise (оба калиброваны
    # к 90%, но разных величин: 90% траекторий vs 90% точек).
    cal = _block(1); test = _block(2)
    _, w_sim = simultaneous_conformal_coverage(*cal, *test, level=0.90)
    _, w_pw = split_conformal_coverage(*cal, *test, level=0.90)
    assert w_sim > w_pw


def test_one_bad_point_breaks_trajectory_coverage():
    pred = np.zeros((1, 10)); std = np.full((1, 10), 0.1); mask = np.ones((1, 10))
    true = np.zeros((1, 10)); true[0, 5] = 5.0                 # один сильный выброс
    cal = (np.zeros((50, 10)), np.full((50, 10), 0.1),
           np.random.default_rng(0).normal(0, 0.1, (50, 10)), np.ones((50, 10)))
    cov, _ = simultaneous_conformal_coverage(*cal, pred, std, true, mask, level=0.90)
    assert cov == 0.0                                          # траектория не покрыта (есть выброс)
