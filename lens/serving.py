"""Shipping a probe as a serving-time metric.

A trained probe is 2 numbers per hidden dim plus a bias. Scoring a token is one
dot product: O(d) against O(d^2) for the layer that produced the activation, so
for d=4096 the probe is ~0.02% of that block's FLOPs -- free, in practice.

The bundle carries the numbers a serving stack needs to use it *correctly*:
which layer to tap, which pooling, the operating threshold, and -- the point of
this repo -- which quantization config the probe was calibrated for, plus the
per-config affine correction so one probe can serve several stacks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ProbeBundle:
    w: np.ndarray
    b: float
    layer: int
    task: str
    pooling: str = "last"
    threshold: float = 0.5
    trained_on: str = "fp16"
    model: str = ""
    # spec name -> (a, c): logit' = a * logit + c, from recalibrate_match.
    corrections: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def score(self, h: np.ndarray, spec: str = None) -> np.ndarray:
        """h: (..., d) residual stream -> (...,) probability."""
        z = np.asarray(h, dtype=np.float32) @ self.w + self.b
        if spec and spec in self.corrections:
            a, c = self.corrections[spec]
            z = a * z + c
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def flag(self, h: np.ndarray, spec: str = None) -> np.ndarray:
        return self.score(h, spec) >= self.threshold

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p.with_suffix(".npz"), w=np.asarray(self.w, dtype=np.float32),
                 b=np.asarray([self.b], dtype=np.float32))
        meta = {k: v for k, v in self.__dict__.items() if k not in ("w", "b")}
        meta["b"] = float(self.b)
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        return p

    @classmethod
    def load(cls, path) -> "ProbeBundle":
        p = Path(path)
        arr = np.load(p.with_suffix(".npz"))
        meta = json.loads(p.with_suffix(".json").read_text())
        meta.pop("b", None)
        return cls(w=arr["w"], b=float(arr["b"][0]), **meta)


def bundle_from_probe(probe, layer: int, task: str, **kw) -> ProbeBundle:
    return ProbeBundle(w=np.asarray(probe.w, dtype=np.float32), b=float(probe.b),
                       layer=layer, task=task, **kw)
