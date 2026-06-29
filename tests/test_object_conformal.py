"""
Тест inductive split-conformal покрытия (#5, исправлено): калибровка на ОТДЕЛЬНОМ наборе,
без transductive утечки ответов теста.
"""
import numpy as np

from liquefaction_ai.evaluation.metrics import split_conformal_coverage


def test_split_conformal_hits_nominal_same_distribution():
    rng = np.random.default_rng(0)
    N, T, sigma = 400, 30, 0.1
    def block(seed):
        r = np.random.default_rng(seed)
        pred = np.zeros((N, T)); std = np.full((N, T), sigma)
        true = pred + r.normal(0, sigma, size=(N, T))
        return pred, std, true, np.ones((N, T))
    cal = block(1); test = block(2)
    cov, w = split_conformal_coverage(*cal, *test, level=0.90)
    assert 0.86 <= cov <= 0.94 and w > 0   # калибровка на cal → номинал на test


def test_split_conformal_does_not_peek_at_test_labels():
    # если калибровочный std занижен, покрытие на тесте падает — т.е. квантиль берётся ТОЛЬКО из cal,
    # а не подгоняется под истинные ошибки теста (что дало бы ~номинал «бесплатно»).
    rng = np.random.default_rng(0)
    N, T = 300, 20
    cal_pred = np.zeros((N, T)); cal_std = np.full((N, T), 0.02)   # заниженный σ на калибровке
    cal_true = cal_pred + rng.normal(0, 0.02, (N, T))
    test_pred = np.zeros((N, T)); test_std = np.full((N, T), 0.02)
    test_true = test_pred + rng.normal(0, 0.10, (N, T))           # тест шумнее
    cov, _ = split_conformal_coverage(cal_pred, cal_std, cal_true, np.ones((N, T)),
                                      test_pred, test_std, test_true, np.ones((N, T)), level=0.90)
    assert cov < 0.5   # недопокрытие, потому что q из cal не знает про шум теста (не transductive)
