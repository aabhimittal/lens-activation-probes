# LENS — activation probes as serving-time metrics

**Question:** a linear probe trained on FP16 activations is cheap enough to run
on every token in production. Does it still mean the same thing when the model
underneath is served with INT4 weights or a 3-bit KV cache?

Production serving quantizes by default. Interpretability research mostly does
not. Nobody has published the crossing point, and the answer decides whether a
probe trained in a research notebook can be trusted as a live metric.

LENS is a small, fully reproducible harness for measuring that gap: it trains
probes on FP16 residual-stream activations, re-extracts the same prompts under
a grid of simulated serving configs, and reports exactly how much of the probe
survives — and which repairs bring it back.

```bash
pip install -e ".[dev]"        # numpy + pyyaml. No GPU, no downloads.
./scripts/run_demo.sh          # full sweep + a serving bundle, ~2 min on 4 cores
pytest -q                      # 44 tests, including the end-to-end sweep
```

For a real model:

```bash
pip install -e ".[hf]"
lens sweep --model Qwen/Qwen3-4B --layers 8 16 24 --out results/qwen3
lens sweep --config experiments/configs/llama-3.2-1b.yaml --out results/llama
```

## Mechanism

1. Tap the residual stream `h_L` at layer `L` inside the forward pass.
2. Score every token with `s = σ(wᵀh_L + b)`. That is `O(d)` against `O(d²)` for
   the layer that produced it — for `d=4096`, under 0.02% of the block. Free.
3. Train probes for hallucination risk, refusal, and injected-instruction
   compliance on FP16 activations.
4. Re-run the *same prompts* under each serving config and measure what moved.
5. Test three repairs, in increasing order of what they cost you.

**Analogy.** A model's self-reported explanation is asking a patient how they
feel. Reading activations is a stethoscope. Quantization swaps in a slightly
different heart mid-checkup — and the question is whether the same stethoscope
reading still means the same thing. The finding below is that the *rhythm*
survives (ranking) while the *absolute reading* drifts (threshold), which is
precisely the failure mode a rank-based metric like AUROC is blind to.

## The measurement that matters, and why AUROC alone misleads

Quantization can break a probe in three different ways:

| failure | what moves | visible in AUROC? | repairable by |
|---|---|---|---|
| rank degradation | class separation | yes | refit only |
| **threshold drift** | score distribution | **no — AUROC is rank-based** | affine recalibration |
| direction rotation | the decodable subspace | yes | refit only |

So every table here reports **flip rate** — the fraction of test decisions that
change at the deployed threshold — next to every AUROC. A team tracking only
AUROC will conclude a probe is robust while its alert volume quietly doubles.

## Results (synthetic backend, CPU, reproducible with `./scripts/run_demo.sh`)

<!-- RESULTS -->

Read `results/synthetic/report.md` for the full tables (per task, per layer, per
probe family). The headline pattern, and the caveat that governs it:

* **Ranking is robust; thresholds are not.** Through INT4 weights and a 4-bit KV
  cache, ΔAUROC stays inside the noise floor, while several percent of decisions
  flip at a fixed threshold. The operational risk is miscalibration, not
  discrimination.
* **3-bit is where it actually breaks.** `w3-g128` is the one config that
  degrades ranking significantly, and it is also the one where affine
  recalibration stops helping — the signature of rotation rather than shift.
* **Weight quantization hurts the probe more than KV quantization** at matched
  bit width, which is the opposite of the intuition that a coarse cache is the
  scarier knob.
* **The cheap repair is the label-free one.** Fitting two parameters `(a, c)` so
  quantized logits reproduce FP16 logits on paired activations needs no labels,
  one offline pass, and folds into `(w, b)` at zero run-time cost.

These numbers come from an 8-layer, d=256 transformer, not a frontier model. The
*pipeline* is what transfers; run `--model` for numbers that mean something about
a specific model. The synthetic backend is a genuine transformer with a real KV
cache and realistic outlier channels — not a mock that fakes degradation with
noise — which is why the ordering (more bits better, smaller groups better,
weights+cache worst) reproduces.

## Serving configs in the grid

`fp16` · `w8-g128` · `w4-g128` · `w4-g32` · `nf4-g64` (QLoRA codebook) ·
`w3-g128` · `kv8` · `kv4` · `kv3` · `w4-g128+kv{8,4,3}` · `w8a8`

Weight quantization is group-wise along input channels (GPTQ/AWQ layout); KV
quantization is per-token-per-head (the vLLM cache layout). Both are *simulated*:
values are quantized to the real grid and dequantized, which reproduces the
serving numerics but not a specific kernel's accumulation order. See
[docs/methodology.md](docs/methodology.md).

## Repairs

| repair | needs | fixes | cannot fix |
|---|---|---|---|
| `match` | unlabeled prompts through both stacks | shift, scale, threshold, calibration | ranking, rotation |
| `affine` | a few dozen labeled quantized examples | the above, plus prior shift | ranking, rotation |
| `refit` | full labeled set + quantized extraction | anything linear | anything non-linear |

## Shipping a probe

`lens export` writes a bundle: the weights, the layer, the pooling, the
threshold, the config it was calibrated on, and a per-config affine correction —
so one probe can serve several stacks.

```python
from lens.serving import ProbeBundle
b = ProbeBundle.load("artifacts/injection_l5")
b.flag(h, spec="w4-g128+kv4")   # correction folded in, still one dot product
```

Getting that dot product inside a real serving stack is the hard part, and it is
not a hook: vLLM captures decode into CUDA graphs, and a Python forward hook
either stops firing on replay (scoring stale activations, silently) or forces a
graph break. [docs/vllm_integration.md](docs/vllm_integration.md) gives the
custom-op version, the graph-safety rules, and the FLOP accounting.

## Layout

```
lens/quant.py          simulated weight / KV / activation quantization + the spec grid
lens/probes.py         logistic and mean-difference probes, and the three repairs
lens/metrics.py        AUROC, AUPRC, ECE, paired-bootstrap CIs, drift metrics
lens/backends/         synthetic numpy transformer (CI) and HF transformers (real models)
lens/experiment.py     the sweep protocol
lens/serving.py        the deployable probe bundle
lens/report.py         markdown tables, computed headline numbers
docs/                  methodology and the vLLM integration path
```

## Weaknesses, stated upfront

* **Correlation, not causation.** A probe finds a direction that *predicts* a
  label; it does not show the model uses that direction to produce the
  behaviour. This is a monitoring tool, and it is not evidence about circuits.
  It should not be cited as one.
* **Label drift.** Probe labels go stale under distribution shift like any
  classifier. Flip rate against a frozen reference set is a canary, not a fix.
* **Proxy labels.** The bundled hallucination task labels *prompts that invite
  fabrication*, not verified fabrications. Swap in real data via `load_jsonl`
  before quoting an absolute number.
* **Templated data.** The built-in datasets are template-generated, which
  inflates absolute AUROC. Splits are template-disjoint by default to blunt
  this, and the quantity of interest is a *relative* delta, which is far less
  sensitive to dataset easiness.
* **Simulated quantization.** Correct grids, not real kernels. Direction and
  rough magnitude transfer; the third decimal does not.
* **Per-architecture plumbing.** The HF backend locates decoder blocks by
  attribute name and covers the Llama/Qwen/GPT-NeoX/OPT layouts; anything else
  needs a line added.

## License

MIT
