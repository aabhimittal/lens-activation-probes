"""Simulated ("fake") quantization of weights and KV-cache tensors.

Everything here is *simulated*: values are quantized to a lower-bit grid and
immediately dequantized back to float. That is deliberate. LENS measures what
quantization does to the *numerics* of the residual stream, not how fast a
kernel runs. A real INT4 kernel and a correctly simulated INT4 grid produce the
same values up to kernel-level accumulation order, so the probe-degradation
numbers transfer; the wall-clock numbers do not.

The ops are written against a tiny array-namespace shim so the exact same code
path runs on numpy arrays (synthetic backend, tests) and torch tensors
(HF backend, GPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# NF4 codebook from QLoRA (Dettmers et al., 2023): 16 levels, normal-float
# spaced so that a unit-variance Gaussian puts equal mass in each bin.
NF4_CODES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


class _NumpyNS:
    @staticmethod
    def imports():
        import numpy as np
        return np

    @classmethod
    def amax(cls, x, axis, keepdims=True):
        return cls.imports().max(x, axis=axis, keepdims=keepdims)

    @classmethod
    def amin(cls, x, axis, keepdims=True):
        return cls.imports().min(x, axis=axis, keepdims=keepdims)

    @classmethod
    def abs(cls, x):
        return cls.imports().abs(x)

    @classmethod
    def clip(cls, x, lo, hi):
        return cls.imports().clip(x, lo, hi)

    @classmethod
    def round(cls, x):
        return cls.imports().round(x)

    @classmethod
    def maximum(cls, x, v):
        return cls.imports().maximum(x, v)

    @classmethod
    def minimum(cls, x, v):
        return cls.imports().minimum(x, v)

    @classmethod
    def concat(cls, xs, axis):
        return cls.imports().concatenate(xs, axis=axis)

    @classmethod
    def bucketize(cls, x, boundaries):
        np = cls.imports()
        return np.searchsorted(np.asarray(boundaries, dtype=x.dtype), x)

    @classmethod
    def gather_codes(cls, idx, codes, like):
        np = cls.imports()
        return np.asarray(codes, dtype=like.dtype)[idx]


class _TorchNS:
    @staticmethod
    def imports():
        import torch
        return torch

    @classmethod
    def amax(cls, x, axis, keepdims=True):
        return x.amax(dim=axis, keepdim=keepdims)

    @classmethod
    def amin(cls, x, axis, keepdims=True):
        return x.amin(dim=axis, keepdim=keepdims)

    @classmethod
    def abs(cls, x):
        return x.abs()

    @classmethod
    def clip(cls, x, lo, hi):
        return x.clamp(lo, hi)

    @classmethod
    def round(cls, x):
        return x.round()

    @classmethod
    def maximum(cls, x, v):
        return x.clamp_min(v)

    @classmethod
    def minimum(cls, x, v):
        return x.clamp_max(v)

    @classmethod
    def concat(cls, xs, axis):
        return cls.imports().cat(xs, dim=axis)

    @classmethod
    def bucketize(cls, x, boundaries):
        torch = cls.imports()
        b = torch.tensor(boundaries, dtype=x.dtype, device=x.device)
        return torch.bucketize(x, b)

    @classmethod
    def gather_codes(cls, idx, codes, like):
        torch = cls.imports()
        c = torch.tensor(codes, dtype=like.dtype, device=like.device)
        return c[idx]


def _ns(x):
    return _TorchNS if type(x).__module__.split(".")[0] == "torch" else _NumpyNS


def _group(x, group_size: int):
    """Reshape the last axis into (n_groups, group_size), padding by edge repeat.

    Edge padding repeats the final column, so it can never widen a group's
    min/max range and therefore never changes the scale of the real elements.
    """
    ns = _ns(x)
    d = x.shape[-1]
    if group_size is None or group_size <= 0 or group_size >= d:
        return x.reshape(*x.shape[:-1], 1, d), d, 0
    pad = (-d) % group_size
    if pad:
        tail = x[..., -1:]
        reps = [tail] * pad
        x = ns.concat([x] + reps, axis=-1)
    n_groups = x.shape[-1] // group_size
    return x.reshape(*x.shape[:-1], n_groups, group_size), group_size, pad


def _ungroup(xg, orig_shape, pad):
    flat = xg.reshape(*orig_shape[:-1], -1)
    if pad:
        flat = flat[..., : orig_shape[-1]]
    return flat


def fake_quant(x, bits: int, group_size: int = -1, scheme: str = "affine"):
    """Quantize-dequantize `x` along its last axis.

    scheme:
      "affine"    - asymmetric min/max (zero-point) grid, the vLLM/AWQ default
      "symmetric" - absmax grid, no zero point (GPTQ-style symmetric)
      "nf4"       - 4-bit normal-float codebook, absmax-scaled (QLoRA)
    """
    if bits is None or bits >= 16:
        return x
    ns = _ns(x)
    orig = x.shape
    xg, gs, pad = _group(x, group_size)

    if scheme == "nf4":
        if bits != 4:
            raise ValueError("nf4 scheme is defined for 4 bits only")
        scale = ns.maximum(ns.amax(ns.abs(xg), -1), 1e-12)
        xn = ns.clip(xg / scale, -1.0, 1.0)
        bounds = [(NF4_CODES[i] + NF4_CODES[i + 1]) / 2 for i in range(len(NF4_CODES) - 1)]
        idx = ns.bucketize(xn, bounds)
        q = ns.gather_codes(idx, NF4_CODES, xn) * scale
    elif scheme == "symmetric":
        qmax = 2 ** (bits - 1) - 1
        scale = ns.maximum(ns.amax(ns.abs(xg), -1) / qmax, 1e-12)
        q = ns.clip(ns.round(xg / scale), -qmax - 1, qmax) * scale
    elif scheme == "affine":
        qmax = 2**bits - 1
        lo = ns.minimum(ns.amin(xg, -1), 0.0)
        hi = ns.maximum(ns.amax(xg, -1), 0.0)
        scale = ns.maximum((hi - lo) / qmax, 1e-12)
        zp = ns.round(-lo / scale)
        q = (ns.clip(ns.round(xg / scale) + zp, 0, qmax) - zp) * scale
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return _ungroup(q, orig, pad)


@dataclass(frozen=True)
class QuantSpec:
    """One serving configuration.

    `weight_*` describes how linear-layer weights are stored; `kv_*` describes
    how the attention KV cache is stored. `act_bits` covers activation
    quantization (W8A8-style) and is applied to the residual stream input of
    each block.
    """
    name: str = "fp16"
    weight_bits: Optional[int] = None
    weight_group: int = 128
    weight_scheme: str = "affine"
    kv_bits: Optional[int] = None
    kv_group: int = -1          # -1 = per-token-per-head (the vLLM default shape)
    kv_scheme: str = "affine"
    act_bits: Optional[int] = None
    act_group: int = -1
    act_scheme: str = "symmetric"
    tags: tuple = field(default_factory=tuple)

    @property
    def is_baseline(self) -> bool:
        return self.weight_bits is None and self.kv_bits is None and self.act_bits is None

    def quant_weight(self, w):
        if self.weight_bits is None:
            return w
        return fake_quant(w, self.weight_bits, self.weight_group, self.weight_scheme)

    def quant_kv(self, t):
        if self.kv_bits is None:
            return t
        return fake_quant(t, self.kv_bits, self.kv_group, self.kv_scheme)

    def quant_act(self, t):
        if self.act_bits is None:
            return t
        return fake_quant(t, self.act_bits, self.act_group, self.act_scheme)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


def _s(**kw) -> QuantSpec:
    return QuantSpec(**kw)


#: The sweep grid. Chosen to separate three questions that get conflated:
#: (a) weight-only quantization, (b) KV-cache-only quantization, (c) both.
STANDARD_SPECS: list[QuantSpec] = [
    _s(name="fp16", tags=("baseline",)),
    _s(name="w8-g128", weight_bits=8, weight_group=128, tags=("weight",)),
    _s(name="w4-g128", weight_bits=4, weight_group=128, tags=("weight",)),
    _s(name="w4-g32", weight_bits=4, weight_group=32, tags=("weight",)),
    _s(name="nf4-g64", weight_bits=4, weight_group=64, weight_scheme="nf4", tags=("weight",)),
    _s(name="w3-g128", weight_bits=3, weight_group=128, tags=("weight", "aggressive")),
    _s(name="kv8", kv_bits=8, tags=("kv",)),
    _s(name="kv4", kv_bits=4, tags=("kv",)),
    _s(name="kv3", kv_bits=3, tags=("kv", "aggressive")),
    _s(name="w4-g128+kv8", weight_bits=4, weight_group=128, kv_bits=8, tags=("combined",)),
    _s(name="w4-g128+kv4", weight_bits=4, weight_group=128, kv_bits=4, tags=("combined",)),
    _s(name="w4-g128+kv3", weight_bits=4, weight_group=128, kv_bits=3,
       tags=("combined", "aggressive")),
    _s(name="w8a8", weight_bits=8, weight_group=128, act_bits=8, tags=("activation",)),
]

SPECS_BY_NAME = {s.name: s for s in STANDARD_SPECS}


def get_specs(names: Optional[list[str]] = None) -> list[QuantSpec]:
    if not names:
        return list(STANDARD_SPECS)
    out = []
    for n in names:
        if n not in SPECS_BY_NAME:
            raise KeyError(f"unknown quant spec {n!r}; have {sorted(SPECS_BY_NAME)}")
        out.append(SPECS_BY_NAME[n])
    return out
