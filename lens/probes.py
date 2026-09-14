"""Linear probes on residual-stream activations.

Two probe families, deliberately:

* LogisticProbe  - L2-regularized logistic regression. What everyone means by
  "linear probe". Highest AUROC, but it can key on low-variance directions,
  which is exactly where quantization noise lives.
* MeanDiffProbe  - difference-of-means direction (optionally whitened, i.e. LDA).
  Weaker on paper, but the direction is a high-variance, low-frequency feature
  of the class structure, so it is the natural candidate to survive a coarser
  numeric grid. Whether it actually does is an empirical question this repo
  answers rather than assumes.

Both cost O(d) per token at serving time: one dot product against the residual
stream, versus O(d^2) for the layer that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metrics import classification_report


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))),
                    np.exp(np.clip(z, -60, 60)) / (1.0 + np.exp(np.clip(z, -60, 60))))


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        return cls(X.mean(0), np.maximum(X.std(0), 1e-8))

    def __call__(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std


class LinearProbe:
    """Common surface: w, b in *raw activation space* so the probe can be
    shipped as a single dot product with no preprocessing at serving time."""

    def __init__(self):
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def decision(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(self.decision(X))

    def evaluate(self, X, y) -> dict:
        return classification_report(y, self.predict_proba(X))

    def save(self, path):
        np.savez(path, w=self.w, b=np.array([self.b]), kind=np.array([type(self).__name__]))

    @staticmethod
    def load(path) -> "LinearProbe":
        d = np.load(path, allow_pickle=False)
        p = LinearProbe()
        p.w, p.b = d["w"], float(d["b"][0])
        return p


class LogisticProbe(LinearProbe):
    def __init__(self, l2: float = 1e-2, steps: int = 400, lr: float = 0.1, seed: int = 0):
        super().__init__()
        self.l2, self.steps, self.lr, self.seed = l2, steps, lr, seed

    def fit(self, X, y) -> "LogisticProbe":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        sc = Standardizer.fit(X)
        Z = sc(X)
        n, d = Z.shape
        w = np.zeros(d)
        b = 0.0
        # Full-batch Adam: deterministic, no scipy dependency, and d is small
        # enough (<= 8k) that full-batch gradients are cheap.
        m = np.zeros(d); v = np.zeros(d); mb = 0.0; vb = 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.steps + 1):
            p = sigmoid(Z @ w + b)
            r = p - y
            gw = Z.T @ r / n + self.l2 * w
            gb = r.mean()
            m = b1 * m + (1 - b1) * gw; v = b2 * v + (1 - b2) * gw**2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb**2
            mh = m / (1 - b1**t); vh = v / (1 - b2**t)
            w -= self.lr * mh / (np.sqrt(vh) + eps)
            b -= self.lr * (mb / (1 - b1**t)) / (np.sqrt(vb / (1 - b2**t)) + eps)
        # Fold the standardizer into (w, b) -> raw-space linear probe.
        self.w = w / sc.std
        self.b = float(b - (w * sc.mean / sc.std).sum())
        return self


class MeanDiffProbe(LinearProbe):
    def __init__(self, whiten: bool = True, shrinkage: float = 0.05):
        super().__init__()
        self.whiten, self.shrinkage = whiten, shrinkage

    def fit(self, X, y) -> "MeanDiffProbe":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
        w = mu1 - mu0
        if self.whiten:
            Xc = np.concatenate([X[y == 1] - mu1, X[y == 0] - mu0], 0)
            cov = Xc.T @ Xc / max(len(Xc) - 2, 1)
            cov += self.shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
            w = np.linalg.solve(cov, w)
        w = w / (np.linalg.norm(w) + 1e-12)
        # 1-D logistic fit on the projection: gives calibrated probabilities
        # without changing the direction.
        z = X @ w
        a, c = 1.0, -float(z.mean())
        for _ in range(200):
            p = sigmoid(a * z + c)
            r = p - y
            ga, gc = float((r * z).mean()), float(r.mean())
            h = np.maximum(p * (1 - p), 1e-6)
            a -= 0.5 * ga / (float((h * z * z).mean()) + 1e-8)
            c -= 0.5 * gc / (float(h.mean()) + 1e-8)
        self.w, self.b = a * w, c
        return self


PROBE_TYPES = {"logistic": LogisticProbe, "meandiff": MeanDiffProbe}


def make_probe(kind: str, **kw) -> LinearProbe:
    if kind not in PROBE_TYPES:
        raise KeyError(f"unknown probe {kind!r}; have {sorted(PROBE_TYPES)}")
    return PROBE_TYPES[kind](**kw)


# ------------------------------------------------------------ repairs ------
def recalibrate_affine(probe: LinearProbe, X_cal, y_cal) -> LinearProbe:
    """Fit scalar (a, c) on the frozen probe's logits: logit' = a*logit + c.

    Two parameters, so a few dozen *labeled* calibration examples on the
    quantized stack are enough. Fixes shift/scale, not rotation.
    """
    z = probe.decision(X_cal)
    y = np.asarray(y_cal, dtype=np.float64).ravel()
    a, c = 1.0, 0.0
    for _ in range(300):
        p = sigmoid(a * z + c)
        r = p - y
        h = np.maximum(p * (1 - p), 1e-6)
        a -= 0.5 * float((r * z).mean()) / (float((h * z * z).mean()) + 1e-8)
        c -= 0.5 * float(r.mean()) / (float(h.mean()) + 1e-8)
    out = LinearProbe()
    out.w, out.b = a * probe.w, a * probe.b + c
    return out


def recalibrate_match(probe: LinearProbe, X_fp, X_q) -> LinearProbe:
    """Label-free repair: fit (a, c) so the quantized logits reproduce the FP
    logits on *paired* activations from the same prompts.

    This is the deployment-friendly variant. You never need new labels, only a
    handful of prompts run through both stacks once, offline.
    """
    zf = probe.decision(X_fp)
    zq = probe.decision(X_q)
    A = np.stack([zq, np.ones_like(zq)], 1)
    a, c = np.linalg.lstsq(A, zf, rcond=None)[0]
    out = LinearProbe()
    out.w, out.b = a * probe.w, a * probe.b + float(c)
    return out


def refit_on_quantized(kind: str, X_q, y, **kw) -> LinearProbe:
    """Upper bound on repair: train the probe directly on quantized activations."""
    return make_probe(kind, **kw).fit(X_q, y)
