"""Classification metrics and FP-vs-quantized drift metrics (numpy only)."""
from __future__ import annotations

import numpy as np


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC with correct tie handling (== Mann-Whitney U / n0 n1)."""
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    n1 = int(y.sum())
    n0 = y.size - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    # Mid-ranks for tie blocks, vectorized: this sits in the inner loop of every
    # bootstrap, so the obvious Python loop over tie blocks is too slow.
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    stop = np.cumsum(counts)
    start = stop - counts
    mid = (start + stop - 1) / 2.0 + 1.0
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = mid[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (step-wise, no interpolation)."""
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / y.sum())


def accuracy(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> float:
    return float(((np.asarray(s) >= threshold).astype(int) == np.asarray(y).astype(int)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def bootstrap_ci(y, s, fn=auroc, n: int = 500, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI. Resamples examples, not tokens."""
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, y.size, y.size)
        if 0 < y[i].sum() < y[i].size:
            vals.append(fn(y[i], s[i]))
    if not vals:
        return (float("nan"), float("nan"))
    v = np.asarray(vals)
    return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))


def paired_delta_ci(y, p_fp, p_q, n: int = 400, alpha: float = 0.05, seed: int = 0):
    """Percentile CI on (AUROC_quantized - AUROC_fp16) under a *paired* bootstrap.

    The two scores come from the same examples, so their errors are strongly
    correlated. Comparing two marginal CIs would be far too conservative and
    would call almost nothing significant. Resampling examples once and
    recomputing both AUROCs on the same resample is the right test.
    """
    y = np.asarray(y).ravel()
    p_fp = np.asarray(p_fp).ravel()
    p_q = np.asarray(p_q).ravel()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, y.size, y.size)
        if 0 < y[i].sum() < y[i].size:
            vals.append(auroc(y[i], p_q[i]) - auroc(y[i], p_fp[i]))
    if not vals:
        return (float("nan"), float("nan"))
    v = np.asarray(vals)
    return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))


def classification_report(y, p, threshold: float = 0.5) -> dict:
    return {
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "acc": accuracy(y, p, threshold),
        "ece": ece(y, p),
        "brier": brier(y, p),
        "pos_rate": float(np.mean(np.asarray(y))),
        "n": int(np.asarray(y).size),
    }


# ---------------------------------------------------------------- drift ----
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def activation_drift(h_fp: np.ndarray, h_q: np.ndarray) -> dict:
    """How far the residual stream itself moved, before any probe is applied."""
    d = h_q - h_fp
    return {
        "cos_mean": float(cosine(h_fp, h_q).mean()),
        "rel_l2": float(np.linalg.norm(d) / (np.linalg.norm(h_fp) + 1e-12)),
        "norm_ratio": float(
            np.linalg.norm(h_q, axis=-1).mean() / (np.linalg.norm(h_fp, axis=-1).mean() + 1e-12)
        ),
    }


def score_drift(p_fp: np.ndarray, p_q: np.ndarray, threshold: float = 0.5) -> dict:
    """How far the *probe's output* moved. Flip rate is the operational number:
    it is the fraction of decisions that change if you swap the serving stack
    underneath a deployed threshold."""
    p_fp = np.asarray(p_fp, dtype=np.float64).ravel()
    p_q = np.asarray(p_q, dtype=np.float64).ravel()
    if p_fp.std() < 1e-12 or p_q.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(p_fp, p_q)[0, 1])
    return {
        "pearson_r": r,
        "mean_shift": float((p_q - p_fp).mean()),
        "mean_abs_shift": float(np.abs(p_q - p_fp).mean()),
        "flip_rate": float(((p_fp >= threshold) != (p_q >= threshold)).mean()),
    }
