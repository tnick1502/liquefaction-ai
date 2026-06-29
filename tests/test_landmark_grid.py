"""
Тест общего early-cycle grid (#6): первые k узлов сетки одинаковы для всех опытов, разрешение
ранней динамики не зависит от N_max.
"""
import numpy as np

from liquefaction_ai.data.ppr_envelope import landmark_aware_cycles


def test_landmark_grid_common_early_resolution():
    k, seq, N0 = 12, 72, 20.0
    # разные опыты с РАЗНЫМ N_max
    g1 = landmark_aware_cycles(last_cycle=100.0, seq_len=seq, landmark_cycles=N0, k_early=k)
    g2 = landmark_aware_cycles(last_cycle=5000.0, seq_len=seq, landmark_cycles=N0, k_early=k)
    # первые k узлов ИДЕНТИЧНЫ (общее раннее разрешение)
    assert np.allclose(g1[:k], g2[:k], atol=1e-4)
    # ранние узлы покрывают [1, N0]; поздние расходятся (зависят от N_max)
    assert abs(g1[0] - 1.0) < 1e-3 and abs(g1[k - 1] - N0) < 1e-2
    assert g1[-1] < g2[-1]                      # поздняя часть длиннее у большего N_max
    assert np.all(np.diff(g1) >= -1e-6)         # монотонность


def test_landmark_grid_points_within_N0_is_fixed():
    k, seq, N0 = 12, 72, 20.0
    for last in (50.0, 500.0, 9000.0):
        g = landmark_aware_cycles(last, seq, N0, k)
        assert int((g <= N0 + 1e-6).sum()) == k   # всегда ровно k точек до N0 (не зависит от N_max)
