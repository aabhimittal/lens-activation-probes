# Running a probe inside a real serving stack

The pipeline in this repo taps activations with PyTorch forward hooks. That is
the right tool for research and the wrong tool for production, and the gap is
the main engineering cost of this project. This document says exactly why, and
what to do instead.

## Why hooks do not survive contact with vLLM

vLLM captures the decode path into CUDA graphs. A CUDA graph is a replayable
recording of kernel launches: it removes per-launch CPU overhead, which is most
of the win at small batch sizes. The recording only contains device work.

A Python `register_forward_hook` is host code. During capture it either runs
once at capture time and is then *never called again on replay* — so your probe
silently scores stale activations forever — or it forces a graph break and you
lose the speedup you were quantizing to get. Both failure modes are quiet. The
first one is worse: the numbers keep flowing and they are wrong.

Related constraints that bite for the same reason:

* the hook cannot allocate (allocation during replay is illegal, and the caching
  allocator's pool is fixed at capture);
* the hook cannot branch on tensor values (no host-visible sync inside a graph);
* anything you write to must be a tensor whose address was fixed at capture.

## What to do instead: a fused op in the forward pass

The probe is a dot product. Put it in the graph.

```python
# lens_op.py -- registered once, called from inside the decoder block
import torch

@torch.library.custom_op("lens::probe_score", mutates_args={"out"})
def probe_score(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                out: torch.Tensor) -> None:
    # h: (num_tokens, d) residual stream; w: (n_probes, d); out: (num_tokens, n_probes)
    # Pre-allocated `out` is the point: no allocation, fixed address, graph-safe.
    torch.addmm(b, h, w.t(), out=out)

@probe_score.register_fake
def _(h, w, b, out):
    return None
```

Call site inside the model definition (vLLM model files are plain PyTorch):

```python
hidden_states = residual + self.mlp(...)
if self.probe_w is not None:                 # static Python bool, decided before capture
    torch.ops.lens.probe_score(hidden_states, self.probe_w, self.probe_b,
                               self.probe_out[:hidden_states.shape[0]])
```

Rules that make this safe under graph capture:

1. **Decide at build time, not at run time.** `if self.probe_w is not None` must
   be a plain Python condition evaluated before capture, never a tensor test.
2. **Pre-allocate `probe_out`** at max batch size in `__init__` and slice it.
   Slicing a pre-allocated tensor keeps the base address stable.
3. **No `.item()`, no `.cpu()`, no printing** inside the op. Thresholding and
   logging happen after the graph replays, on the output buffer.
4. **Batch the probes.** `w` is `(n_probes, d)`, so ten probes cost one GEMM
   with a tiny N dimension, not ten kernel launches.

Cost: `(T, d) @ (d, n_probes)`, i.e. `2·T·d·n_probes` FLOPs against roughly
`2·T·d²·(4 + 3·ffn_ratio)` for the block that produced `h`. For `d=4096` and ten
probes that is under 0.02% of the block. It is free in every sense that matters;
what is not free is the per-architecture plumbing, since the call site lives in
the model definition and every model file is written by hand.

## Placement

Tap the residual stream *after* the block, at the layer your sweep picked
(`lens sweep` reports layer sensitivity). Prefill and decode both flow through
the same op, so you get a score per token with no extra passes. Pooling to a
per-request score (`last` in this repo) is done outside the graph.

## Talking to the quantized stack

This is the part the rest of the repo is about. The probe you trained on FP16
activations is not automatically valid against the deployed config. The
serving-time contract is in `lens/serving.py`: a bundle records the layer, the
pooling, the threshold, the config it was calibrated on, and a per-config affine
correction `(a, c)` applied to the logit. Fold `(a, c)` into `w` and `b` before
capture and the correction costs nothing at run time.

Re-derive the correction whenever the serving config changes — a new quantizer,
a new group size, a new KV dtype. It needs no labels, only a few hundred prompts
run through both stacks once (`recalibrate_match`).
