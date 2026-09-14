import numpy as np

from lens.metrics import (accuracy, activation_drift, auprc, auroc, bootstrap_ci, ece,
                          score_drift)


def test_auroc_known_cases():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(y, np.ones(4)) == 0.5  # all ties -> chance


def test_auroc_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200), 1)  # forces many ties
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert abs(auroc(y, s) - brute) < 1e-9


def test_auprc_and_accuracy():
    y = np.array([0, 1, 1, 0])
    assert abs(auprc(y, np.array([0.1, 0.9, 0.8, 0.2])) - 1.0) < 1e-9
    assert accuracy(y, np.array([0.1, 0.9, 0.8, 0.2])) == 1.0


def test_ece_is_zero_for_perfect_calibration_and_one_for_worst():
    y = np.array([0, 0, 1, 1])
    assert ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert abs(ece(y, np.array([1.0, 1.0, 0.0, 0.0])) - 1.0) < 1e-9


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y * 0.6 + rng.normal(size=300) * 0.5
    lo, hi = bootstrap_ci(y, s, n=200)
    assert lo < auroc(y, s) < hi


def test_score_drift_flags_threshold_crossings():
    p_fp = np.array([0.4, 0.6, 0.45])
    p_q = np.array([0.6, 0.6, 0.44])
    d = score_drift(p_fp, p_q)
    assert abs(d["flip_rate"] - 1 / 3) < 1e-9
    assert d["mean_abs_shift"] > 0


def test_activation_drift_is_zero_for_identical_inputs():
    h = np.random.default_rng(0).normal(size=(16, 8))
    d = activation_drift(h, h)
    assert abs(d["cos_mean"] - 1) < 1e-9 and d["rel_l2"] < 1e-9
