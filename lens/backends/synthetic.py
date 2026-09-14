"""A small but genuine transformer in numpy.

This is not a mock that "simulates degradation" by adding noise. It runs real
matmuls through real weights and a real causal KV cache, so when you quantize
the weights or the cache, the activations move for the same reason they move in
vLLM: the numbers going into the dot products are different.

Why it exists: the quantization-robustness question needs a stack you can run
thousands of times in CI on a CPU, with no download and no GPU. Findings here
are directional only -- a 4-layer d=64 model is not Qwen3-4B -- but the
*pipeline* is identical, and the HF backend is a drop-in replacement.

Two properties are copied from real LLMs because they drive quantization error:
  * outlier feature channels (a handful of residual dims with ~10x the scale),
    which blow up per-group absmax and are why group size matters so much;
  * a heavy-tailed token-embedding spectrum.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..quant import QuantSpec
from .base import pool


def hash_tokenize(text: str, vocab: int, max_len: int = 64) -> list[int]:
    """Deterministic whitespace + hash tokenizer. No dependency, no download."""
    toks = text.lower().replace(",", " ,").replace(".", " .").split()[:max_len]
    ids = []
    for w in toks:
        h = 2166136261
        for ch in w.encode("utf-8"):  # FNV-1a
            h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
        ids.append(h % vocab)
    return ids or [0]


def rmsnorm(x, w, eps=1e-6):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * w


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


class SyntheticBackend:
    name = "synthetic"

    def __init__(self, n_layers: int = 8, d_model: int = 256, n_heads: int = 8,
                 vocab: int = 2048, d_ff: int = 512, seed: int = 0,
                 n_outlier_channels: int = 3, outlier_scale: float = 8.0,
                 max_len: int = 64):
        rng = np.random.default_rng(seed)
        self.n_layers, self.d_model, self.n_heads = n_layers, d_model, n_heads
        self.vocab, self.d_ff, self.max_len = vocab, d_ff, max_len
        self.d_head = d_model // n_heads
        assert self.d_head * n_heads == d_model, "d_model must divide by n_heads"

        scale = 1.0 / np.sqrt(d_model)
        # Heavy-tailed embeddings + explicit outlier channels.
        emb = rng.normal(size=(vocab, d_model)) * scale
        emb *= rng.lognormal(0.0, 0.6, size=(vocab, 1))
        self.outlier_channels = rng.choice(d_model, n_outlier_channels, replace=False)
        emb[:, self.outlier_channels] *= outlier_scale
        self.emb = emb.astype(np.float32)
        self.pos = (rng.normal(size=(max_len, d_model)) * scale * 0.5).astype(np.float32)

        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append({
                # weights stored (out, in) so group-wise quantization groups
                # along the *input* channels, exactly as GPTQ/AWQ do.
                "wq": (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32),
                "wk": (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32),
                "wv": (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32),
                "wo": (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32),
                "w1": (rng.normal(size=(d_ff, d_model)) * scale).astype(np.float32),
                "w2": (rng.normal(size=(d_model, d_ff)) * (1.0 / np.sqrt(d_ff))).astype(np.float32),
                "n1": np.ones(d_model, dtype=np.float32),
                "n2": np.ones(d_model, dtype=np.float32),
            })

    # ------------------------------------------------------------------
    def _quantized_blocks(self, spec: QuantSpec):
        if spec.weight_bits is None:
            return self.blocks
        out = []
        for blk in self.blocks:
            q = dict(blk)
            for k in ("wq", "wk", "wv", "wo", "w1", "w2"):
                q[k] = spec.quant_weight(blk[k].astype(np.float32))
            out.append(q)
        return out

    def _attend(self, x, blk, spec):
        T = x.shape[0]
        H, dh = self.n_heads, self.d_head
        q = (x @ blk["wq"].T).reshape(T, H, dh)
        k = (x @ blk["wk"].T).reshape(T, H, dh)
        v = (x @ blk["wv"].T).reshape(T, H, dh)
        # KV-cache quantization: this is where the cached tensors are *stored*,
        # so quantize here, per (token, head) row -- the layout vLLM's
        # fp8/int8 KV cache actually uses.
        k, v = spec.quant_kv(k), spec.quant_kv(v)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(dh)
        mask = np.triu(np.full((T, T), -1e30, dtype=np.float32), 1)
        att = att + mask
        att -= att.max(-1, keepdims=True)
        p = np.exp(att)
        p /= p.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", p, v).reshape(T, H * dh)
        return o @ blk["wo"].T

    def forward(self, ids: Sequence[int], spec: QuantSpec, blocks=None) -> list[np.ndarray]:
        """-> list of (T, d_model) residual streams, one per layer (post-block)."""
        blocks = self.blocks if blocks is None else blocks
        ids = list(ids)[: self.max_len]
        x = self.emb[np.asarray(ids)] + self.pos[: len(ids)]
        outs = []
        for blk in blocks:
            x = spec.quant_act(x)
            x = x + self._attend(rmsnorm(x, blk["n1"]), blk, spec)
            h = rmsnorm(x, blk["n2"])
            x = x + gelu(h @ blk["w1"].T) @ blk["w2"].T
            outs.append(x)
        return outs

    def activations(self, prompts, layers, spec: QuantSpec, pooling: str = "last"):
        blocks = self._quantized_blocks(spec)
        layers = list(layers)
        acc = {l: [] for l in layers}
        for p in prompts:
            hs = self.forward(hash_tokenize(p, self.vocab, self.max_len), spec, blocks)
            for l in layers:
                acc[l].append(pool(hs[l], pooling))
        return {l: np.asarray(v, dtype=np.float32) for l, v in acc.items()}
