"""
Тесты причинного sustained-onset критерия (замечания рецензента по раунду N+4).

Гарантируют, что критерий:
* работает на СЫРОЙ немонотонной кривой (одиночный всплеск ≠ onset);
* требует ПОЛНОЕ окно (усечённый хвост из 1 записи не засчитывается);
* не превращает укороченный терминальный хвост в событие;
* строг к монотонной кривой не эквивалентен: на монотонной первое пересечение = onset (sustain no-op),
  поэтому критерий ДОЛЖЕН применяться к сырым пикам.
"""
import numpy as np

from liquefaction_ai.config import LIQ_THRESHOLD
from liquefaction_ai.data.raw_loader import sustained_first_crossing as sfc, terminal_onset_ambiguous


def test_single_mid_spike_is_not_onset():
    # одиночный всплеск ≥ порога, затем падение — НЕ устойчивое пересечение
    curve = [0.1, 0.5, 0.97, 0.4, 0.5, 0.6, 0.7]
    assert sfc(curve, thr=LIQ_THRESHOLD, sustain=3) == -1


def test_sustained_window_detects_onset():
    curve = [0.1, 0.5, 0.97, 0.98, 0.99, 0.6, 0.7]
    assert sfc(curve, thr=LIQ_THRESHOLD, sustain=3) == 2


def test_truncated_tail_single_cycle_is_not_onset():
    # единственный последний цикл ≥ порога — не полное окно и не терминальный onset (нужно ≥2)
    curve = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.97]
    assert sfc(curve, thr=LIQ_THRESHOLD, sustain=3) == -1


def test_terminal_onset_two_tail_cycles_is_ambiguous_not_event():
    curve = [0.1, 0.2, 0.3, 0.4, 0.5, 0.96, 0.99]
    assert sfc(curve, thr=LIQ_THRESHOLD, sustain=3) == -1
    assert terminal_onset_ambiguous(curve, thr=LIQ_THRESHOLD, sustain=3)


def test_nonconsecutive_observations_do_not_form_sustained_window():
    curve = [0.1, 0.96, 0.98, 0.99]
    cycles = [1.0, 5.0, 7.0, 8.0]
    assert sfc(curve, thr=LIQ_THRESHOLD, sustain=3, cycles=cycles) == -1


def test_subcycle_phase_jitter_still_consecutive():
    # Режим points_in_cycle: номера циклов пиков дрожат по фазе внутри цикла (k+0.9, (k+1)+0.1...),
    # но это ПОДРЯД идущие целые циклы. Критерий не должен ложно цензурировать onset из-за float-diff.
    curve = [0.10, 0.30, 0.96, 0.98, 0.99]
    cycles = [1.05, 1.95, 2.90, 4.10, 5.05]   # целые циклы 1,1?,2,4,5 → floor: 1,1,2,4,5
    # исправим на реальный дрожащий, но последовательный случай:
    cycles = [1.05, 2.95, 3.90, 4.15, 5.05]   # floor: 1,2,3,4,5 — последовательные
    assert sfc(curve, thr=LIQ_THRESHOLD, sustain=3, cycles=cycles) == 2


def test_no_crossing_is_censored():
    curve = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert sfc(curve, thr=LIQ_THRESHOLD, sustain=3) == -1


def test_sustain_matters_on_raw_but_not_on_monotone():
    # На СЫРОЙ кривой ранний одиночный всплеск игнорируется, onset — на устойчивом участке.
    raw = np.array([0.1, 0.96, 0.5, 0.97, 0.98, 0.99, 0.99])
    assert sfc(raw, thr=LIQ_THRESHOLD, sustain=3) == 3
    # На МОНОТОННОЙ (cummax) кривой первое же пересечение остаётся выше — sustain ничего не меняет:
    mono = np.maximum.accumulate(raw)
    assert sfc(mono, thr=LIQ_THRESHOLD, sustain=3) == 1  # = первое пересечение (демонстрация no-op)
