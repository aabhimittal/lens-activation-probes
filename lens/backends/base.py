"""Backend protocol: anything that can return residual-stream activations."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..quant import QuantSpec

PoolSpec = str  # "last" | "mean" | "max"


def pool(H: np.ndarray, how: PoolSpec) -> np.ndarray:
    """(T, d) token activations -> (d,) example activation."""
    if how == "last":
        return H[-1]
    if how == "mean":
        return H.mean(0)
    if how == "max":
        return H[np.argmax(np.linalg.norm(H, axis=-1))]
    raise ValueError(f"unknown pooling {how!r}")


class Backend(Protocol):
    name: str
    n_layers: int
    d_model: int

    def activations(
        self, prompts: Sequence[str], layers: Sequence[int], spec: QuantSpec,
        pooling: PoolSpec = "last",
    ) -> dict[int, np.ndarray]:
        """-> {layer: (n_prompts, d_model) float32}, under serving config `spec`."""
        ...


def get_backend(kind: str, **kw) -> Backend:
    if kind == "synthetic":
        from .synthetic import SyntheticBackend
        return SyntheticBackend(**kw)
    if kind in ("hf", "transformers"):
        from .hf import HFBackend
        return HFBackend(**kw)
    raise KeyError(f"unknown backend {kind!r}")
