# Methodology, and what these numbers can and cannot support

## The claim under test

A linear probe `s = σ(wᵀh_L + b)` is trained on FP16 residual-stream
activations. Production serves the same model with INT4 weights and/or a
low-bit KV cache. Does the probe still measure what it measured?

Three ways the answer can be "no", and they fail differently:

1. **Rank degradation.** The quantized activations no longer separate the
   classes as well. AUROC drops. This is the failure people look for.
2. **Threshold drift.** Ranking survives, but the score distribution shifts, so
   a deployed threshold now fires at a different rate. AUROC is rank-based and
   *cannot see this at all*. `flip_rate` and `ECE` can.
3. **Direction rotation.** The class-relevant subspace moves. Affine
   recalibration cannot fix a rotation — only refitting can. The gap between
   `flip +match` and `AUROC +refit` is the diagnostic.

Separating (2) from (1) is the reason this repo reports flip rate next to every
AUROC. A team that only tracks AUROC will conclude a probe is robust while its
alert volume quietly doubles.

## Protocol

* Probes are trained on FP16 activations only. Quantized activations are never
  seen at training time, except by the explicit `refit` repair, which exists to
  bound how much is recoverable.
* Train/test splits are **template-disjoint** by default. A probe that memorizes
  a template gets no credit.
* Every spec is run over the *same* prompts, so FP16 and quantized activations
  are paired example-by-example. That pairing is what makes label-free
  recalibration and activation-drift metrics possible.
* The baseline row is asserted to have exactly zero drift; if extraction were
  non-deterministic, every delta in the table would be noise, so the test suite
  checks it.

## Repairs, in increasing order of cost

| repair | needs | fixes | cannot fix |
|---|---|---|---|
| `match` | a few hundred *unlabeled* prompts run through both stacks | shift, scale, threshold, calibration | ranking, rotation |
| `affine` | a few dozen *labeled* examples on the quantized stack | same as above, plus prior shift | ranking, rotation |
| `refit` | full labeled training set + quantized extraction | everything a linear probe can fix | anything non-linear |

`match` is the one to deploy: no labels, one offline pass, and it folds into
`(w, b)` so it costs nothing at run time.

## Known weaknesses

**Correlation, not causation.** A probe finds a direction that *predicts* a
label. It does not show that the model uses that direction to produce the
behaviour. Everything here is observational. It is a monitoring tool, not a
mechanistic claim, and it should not be cited as evidence about circuits. A
probe can ride on a confound — prompt length, formatting, topic — and stay
accurate until the confound moves.

**Label drift.** Probe labels go stale under distribution shift like any
classifier. "Hallucination risk" defined over one prompt distribution is not the
same target three months later. Flip rate against a frozen reference set is a
cheap canary; it is not a fix.

**Simulated quantization.** Weights and cache entries are quantized to the
correct grid and dequantized, rather than stored in low precision. Numerically
this reproduces the serving grid; it does not reproduce a specific kernel's
accumulation order, dynamic activation scaling, or a runtime's calibration
pass. Expect the direction and rough magnitude to transfer, not the third
decimal.

**The synthetic backend is a model of a model.** It is a real transformer with
real KV-cache quantization and realistic outlier channels, which is why it
reproduces the qualitative ordering (more bits better, smaller groups better,
weights + cache worst). It is not evidence about Qwen3-4B. Run `--model` for
that; the code path is identical.

**Proxy labels.** The bundled hallucination task labels *prompts that invite
fabrication*, not verified fabrications in the output. It is a risk proxy.
Substitute a real verified-answer dataset via `load_jsonl` before quoting an
absolute number.
