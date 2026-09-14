import numpy as np
import pytest

from lens.metrics import auroc, ece, score_drift
from lens.probes import (LogisticProbe, MeanDiffProbe, make_probe, recalibrate_affine,
                         recalibrate_match, refit_on_quantized)


def make_data(n=600, d=32, sep=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d)) + np.outer(y, w) * sep
    return X, y


@pytest.mark.parametrize("kind", ["logistic", "meandiff"])
def test_probes_learn_separable_structure(kind):
    X, y = make_data()
    p = make_probe(kind).fit(X[:400], y[:400])
    assert p.evaluate(X[400:], y[400:])["auroc"] > 0.9
    assert p.w.shape == (X.shape[1],)


def test_probe_is_a_single_dot_product_in_raw_space():
    """The standardizer must be folded into (w, b); serving code applies no
    preprocessing."""
    X, y = make_data()
    p = LogisticProbe().fit(X, y)
    assert np.allclose(p.decision(X), X @ p.w + p.b)


def test_unlearnable_data_gives_chance_auroc():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(400, 16)), rng.integers(0, 2, 400)
    p = LogisticProbe().fit(X[:300], y[:300])
    assert 0.3 < p.evaluate(X[300:], y[300:])["auroc"] < 0.7


def test_match_recalibration_preserves_ranking_but_fixes_calibration():
    X, y = make_data()
    p = LogisticProbe().fit(X[:400], y[:400])
    Xq = X * 1.5 + 0.3          # affine corruption, as a scale/shift of the stream
    fixed = recalibrate_match(p, X[:64], Xq[:64])
    a_before = auroc(y[400:], p.predict_proba(Xq[400:]))
    a_after = auroc(y[400:], fixed.predict_proba(Xq[400:]))
    assert abs(a_before - a_after) < 1e-9          # monotone: ranking untouched
    assert ece(y[400:], fixed.predict_proba(Xq[400:])) <= ece(y[400:], p.predict_proba(Xq[400:])) + 1e-9
    assert (score_drift(p.predict_proba(X[400:]), fixed.predict_proba(Xq[400:]))["flip_rate"]
            <= score_drift(p.predict_proba(X[400:]), p.predict_proba(Xq[400:]))["flip_rate"])


def test_affine_recalibration_improves_calibration_with_labels():
    X, y = make_data()
    p = LogisticProbe().fit(X[:400], y[:400])
    Xq = X * 0.5 - 0.4
    fixed = recalibrate_affine(p, Xq[:128], y[:128])
    assert ece(y[400:], fixed.predict_proba(Xq[400:])) < ece(y[400:], p.predict_proba(Xq[400:]))


def test_refit_beats_transfer_under_rotation():
    """A rotation is the corruption affine recalibration provably cannot fix;
    refitting can."""
    X, y = make_data(sep=0.45)
    rng = np.random.default_rng(7)
    Q = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))[0]
    Xq = X @ Q
    p = LogisticProbe().fit(X[:400], y[:400])
    transferred = auroc(y[400:], p.predict_proba(Xq[400:]))
    refit = auroc(y[400:], refit_on_quantized("logistic", Xq[:400], y[:400]).predict_proba(Xq[400:]))
    assert refit > transferred + 0.1


def test_probe_roundtrip(tmp_path):
    X, y = make_data()
    p = MeanDiffProbe().fit(X, y)
    f = tmp_path / "p.npz"
    p.save(f)
    q = MeanDiffProbe.load(f)
    assert np.allclose(q.predict_proba(X), p.predict_proba(X))
