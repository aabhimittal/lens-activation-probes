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
pytest -q                      # 45 tests, including the end-to-end sweep
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

- Mean ΔAUROC across all quantized configs: **-0.005** (mean flip rate at the deployed threshold: **0.060**).
- Worst single cell: `w3-g128` on `hallucination` layer 2 (meandiff), ΔAUROC **-0.134**.
- Weight-only INT4 configs: mean ΔAUROC 0.000, mean flip rate 0.077.
- KV-cache-only configs: mean ΔAUROC 0.002, mean flip rate 0.031.
- 72/288 quantized cells show a ΔAUROC whose paired 95% bootstrap CI excludes zero; the rest are inside the noise floor.
- Repairs: label-free affine matching cuts mean flip rate 0.060 -> 0.056; retraining on quantized activations moves mean AUROC 0.902 -> 0.908.

Logistic probe, averaged over 3 tasks x 4 layers (`*` = paired bootstrap CI on the delta excludes zero):

| serving config | AUROC fp16 | AUROC | ΔAUROC | act rel-L2 | score r | flip rate | ECE | ECE +match | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|---|---|---|---|
| w4-g128+kv3 | 0.915 | 0.919 | 0.004 | 0.368 | 0.915 | 0.076 | 0.256 | 0.246 | 0.092 | 0.912 |
| w4-g32 | 0.915 | 0.917 | 0.002 | 0.234 | 0.949 | 0.060 | 0.292 | 0.283 | 0.053 | 0.908 |
| w8-g128 | 0.915 | 0.917 | 0.002 | 0.017 | 1.000 | 0.000 | 0.262 | 0.261 | 0.000 | 0.916 |
| kv4 | 0.915 | 0.916 | 0.001 | 0.114 | 0.984 | 0.026 | 0.271 | 0.270 | 0.024 | 0.924 |
| fp16 | 0.915 | 0.915 | 0.000 | 0.000 | 1.000 | 0.000 | 0.264 | 0.264 | 0.000 | 0.915 |
| kv8 | 0.915 | 0.915 | 0.000 | 0.007 | 1.000 | 0.002 | 0.261 | 0.262 | 0.002 | 0.916 |
| kv3 | 0.915 | 0.915 | -0.001 | 0.232 | 0.961 | 0.062 | 0.236 | 0.243 | 0.056 | 0.906 |
| w8a8 | 0.915 | 0.914 | -0.001 | 0.030 | 0.999 | 0.004 | 0.263 | 0.261 | 0.000 | 0.914 |
| w4-g128 | 0.915 | 0.912 | -0.003 | 0.284 | 0.922 | 0.072 | 0.262 | 0.251 | 0.083 | 0.922 |
| w4-g128+kv8 | 0.915 | 0.911 | -0.004 | 0.284 | 0.921 | 0.072 | 0.261 | 0.253 | 0.085 | 0.922 |
| w4-g128+kv4 | 0.915 | 0.906 | -0.009 | 0.304 | 0.927 | 0.072 | 0.264 | 0.251 | 0.084 | 0.909 |
| nf4-g64 | 0.915 | 0.902 | -0.013 | 0.262 | 0.942 | 0.061 | 0.284 | 0.254 | 0.045 | 0.904 |
| w3-g128 | 0.915 | 0.867 | -0.048* | 0.566 | 0.842 | 0.163 | 0.228 | 0.260 | 0.130 | 0.845 |

<!-- /RESULTS -->

Read `results/synthetic/report.md` for the full tables (per task, per layer, per
probe family). The headline pattern, and the caveat that governs it:

* **Ranking is robust; thresholds are not.** Through INT4 weights and a 4-bit KV
  cache, ΔAUROC stays inside the noise floor (−0.009, paired CI includes zero),
  while **7% of decisions flip** at a fixed threshold. The operational risk here
  is miscalibration, not discrimination — and AUROC cannot see it.
* **3-bit *weights* are where it actually breaks; 3-bit *cache* is not.**
  `w3-g128` is the only config with a significant ΔAUROC (−0.048, CI excludes
  zero) and 16% of decisions flipping. `kv3`, at the same nominal bit width,
  costs −0.001 AUROC. Weight precision is the binding constraint; cache
  precision is close to free for probes.
* **Weight quantization hurts more than KV quantization at matched bit width**,
  which inverts the usual intuition that a coarse cache is the scarier knob:
  flip rate 0.072 (`w4-g128`) vs 0.026 (`kv4`), and 0.163 (`w3-g128`) vs 0.062
  (`kv3`). Smaller groups help as expected: `w4-g32` flips 0.060 vs 0.072 for
  `w4-g128`.
* **Neither repair is a cure, and the honest version is worth stating.**
  Label-free affine matching needs no labels and costs nothing at run time, but
  it only moves mean flip rate 0.060 → 0.056, and at `w3-g128` it makes ECE
  *worse* (0.228 → 0.260) — a signature that the drift there is rotation, not
  shift, which is exactly what two parameters cannot fix. Refitting on quantized
  activations is also **not** a reliable upper bound: at `w3-g128` it scores
  0.845, *below* the 0.867 of the untouched FP16-trained probe, because a
  direction fit on degraded activations can be worse than one transferred from
  clean ones. The useful conclusion is that the fix for 3-bit weights is not to
  patch the probe.

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
| `refit` | full labeled set + quantized extraction | rotation, rank loss | anything non-linear; can lose to plain transfer at 3-bit |

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
