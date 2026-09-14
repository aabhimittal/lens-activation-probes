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
  estimate how much is recoverable. Note it is an estimate, not a bound: refit
  can score below plain transfer when the quantized activations carry less
  signal than the FP16 direction can still extract from them.
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
| `refit` | full labeled training set + quantized extraction | rotation, rank loss -- the strongest linear repair | anything non-linear, and it can *lose* to plain transfer when the quantized activations are genuinely noisier |

`match` is the one to deploy: no labels, one offline pass, and it folds into
`(w, b)` so it costs nothing at run time.

## Verified hallucination labels

The bundled `hallucination` task is a prompt-level proxy: it labels questions
that presuppose a non-existent entity. `lens label` replaces it with the label
the question deserves.

1. Sample real closed-book questions (TriviaQA `rc.nocontext`, which ships
   normalized answer aliases).
2. Greedy-decode an answer **on the FP16 reference stack**.
3. Grade it: label 1 if no gold alias appears in the normalized output.
4. Balance the classes by subsampling, and report the true base rate.

The probe then predicts, from the residual stream at the **last prompt token**
-- before a single answer token exists -- whether this model is about to get it
wrong. That is a genuine forward-looking signal rather than a property of the
prompt's surface form.

Three design choices worth defending:

* **Labels are generated once, on FP16, and then frozen.** Regenerating them per
  serving config would let the target move with the thing being measured, and
  ΔAUROC would become uninterpretable. The quantized stacks are scored against
  the reference model's behaviour.
* **Grading is lenient** (substring match against any alias). A small model
  answers in a sentence, so exact match would label nearly everything wrong and
  the task would collapse into "did the model emit a bare noun phrase". Lenient
  matching over-credits, which biases *against* finding probe signal -- the safe
  direction to err in.
* **Labels are model-specific.** A label set built against Qwen2.5-0.5B says
  nothing about any other model. The file name records which model produced it,
  and the `.meta.json` alongside records the base rate, the source split, and the
  decode length.

The honest limitation: small models are wrong most of the time, so balancing
throws away most of the majority class and the usable set is much smaller than
the number of questions asked. The base rate is reported rather than buried,
because a probe trained on a balanced subsample of a 15%-accurate model is not
measuring the same thing as one trained on a balanced sample of a strong model.

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
