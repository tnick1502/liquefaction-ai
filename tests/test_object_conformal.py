"""
Тест object-held-out (CV+) conformal покрытия.

Проверяет, что при site-level shift (у разных объектов свой масштаб ошибки) наивная гауссова
калибровка z·σ систематически недопокрывает, а leave-one-object-out conformal поднимает покрытие
к номиналу 0.90.
"""
import numpy as np

from liquefaction_ai.evaluation.metrics import object_conformal_coverage


def test_objconf_beats_gaussian_under_site_shift():
    rng = np.random.default_rng(0)
    n_obj, per, T = 8, 40, 30
    preds, trues, stds, objs = [], [], [], []
    for o in range(n_obj):
        scale = 1.0 + 2.5 * (o / n_obj)        # масштаб ошибки растёт по объектам (site shift)
        p = np.zeros((per, T), dtype=float)
        s = np.full((per, T), 0.1, dtype=float)  # модель рапортует одинаковый недооценённый σ
        t = p + rng.normal(0.0, 0.1 * scale, size=(per, T))
        preds.append(p); stds.append(s); trues.append(t); objs += [f"obj{o}"] * per
    pred = np.vstack(preds); std = np.vstack(stds); true = np.vstack(trues)
    mask = np.ones_like(pred); objects = np.array(objs)

    # наивное гауссово покрытие@90 (z=1.645)
    z = 1.6449
    cov_gauss = float(((true >= pred - z * std) & (true <= pred + z * std)).mean())
    cov_oc, width_oc = object_conformal_coverage(pred, std, true, mask, objects, level=0.90)

    assert cov_gauss < 0.85, f"ожидалось недопокрытие наивной калибровки, получено {cov_gauss}"
    assert 0.86 <= cov_oc <= 0.94, f"object-conformal покрытие вне полосы: {cov_oc}"
    assert width_oc > 0


def test_objconf_nan_with_few_objects():
    pred = np.zeros((4, 5)); std = np.ones((4, 5)); true = np.zeros((4, 5)); mask = np.ones((4, 5))
    cov, w = object_conformal_coverage(pred, std, true, mask, np.array(["a", "a", "b", "b"]))
    assert np.isnan(cov) and np.isnan(w)   # < 3 объектов → не определено
