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
pytest -q                      # 58 tests, including the end-to-end sweep
```

For a real model (CPU is enough for the small ones):

```bash
pip install -e ".[hf]"

# 1. verified hallucination labels: let the model answer real questions,
#    grade it, and label the ones it actually got wrong
lens label --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --dtype float32 \
           --n 1200 --out data/hallucination_qwen2.5-0.5b.jsonl

# 2. sweep the serving grid against those labels
lens sweep --config experiments/configs/qwen2.5-0.5b-cpu.yaml --out results/qwen

# 3. does quantization change WHICH questions the model fails?
python scripts/measure_label_drift.py --model Qwen/Qwen2.5-0.5B-Instruct \
       --device cpu --dtype float32 --n 400
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

## Results on a real model

Qwen2.5-0.5B-Instruct, 10 serving configs x 3 tasks x 4 layers x 2 probe
families, CPU-only and reproducible from the committed config.

<!-- REAL-RESULTS -->

- Mean ΔAUROC across all quantized configs: **-0.037** (mean flip rate at the deployed threshold: **0.128**).
- Worst single cell: `kv3` on `data/hallucination_qwen2.5-0.5b.jsonl` layer 6 (logistic), ΔAUROC **-0.296**.
- Weight-only INT4 configs: mean ΔAUROC -0.034, mean flip rate 0.133.
- KV-cache-only configs: mean ΔAUROC -0.056, mean flip rate 0.157.
- 46/216 quantized cells show a ΔAUROC whose paired 95% bootstrap CI excludes zero; the rest are inside the noise floor.
- Repairs: label-free affine matching cuts mean flip rate 0.128 -> 0.119; retraining on quantized activations moves mean AUROC 0.862 -> 0.885.

Per task, and whether there was any headroom to lose:

| task | probe | AUROC fp16 | headroom | AUROC quantized | ΔAUROC | flip rate | AUROC +refit |
|---|---|---|---|---|---|---|---|
| hallucination_qwen2.5-0.5b | logistic | 0.708 | 0.292 | 0.617 | -0.091 | 0.305 | 0.671 |
| hallucination_qwen2.5-0.5b | meandiff | 0.685 | 0.315 | 0.601 | -0.084 | 0.317 | 0.647 |
| injection | logistic | 1.000 | 0.000 | 0.999 | -0.001 | 0.033 | 0.999 |
| injection | meandiff | 1.000 | 0.000 | 0.999 | -0.001 | 0.006 | 1.000 |
| refusal | logistic | 1.000 | 0.000 | 0.987 | -0.013 | 0.019 | 1.000 |
| refusal | meandiff | 1.000 | 0.000 | 0.989 | -0.011 | 0.014 | 1.000 |

Logistic probe, averaged over 3 tasks x 4 layers (`*` = paired bootstrap CI on the delta excludes zero):

| serving config | AUROC fp16 | AUROC | ΔAUROC | act rel-L2 | score r | flip rate | ECE | ECE +match | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|---|---|---|---|
| w8-g128 | 0.903 | 0.903 | 0.001 | 0.015 | 0.997 | 0.011 | 0.080 | 0.076 | 0.003 | 0.904 |
| fp16 | 0.903 | 0.903 | 0.000 | 0.000 | 1.000 | 0.000 | 0.078 | 0.078 | 0.000 | 0.903 |
| nf4-g64 | 0.903 | 0.900 | -0.003 | 0.202 | 0.867 | 0.106 | 0.117 | 0.074 | 0.058 | 0.901 |
| kv8 | 0.903 | 0.887 | -0.015 | 0.135 | 0.916 | 0.071 | 0.100 | 0.078 | 0.055 | 0.908 |
| w4-g128 | 0.903 | 0.883 | -0.020 | 0.234 | 0.858 | 0.101 | 0.109 | 0.079 | 0.071 | 0.903 |
| w4-g32 | 0.903 | 0.881 | -0.021 | 0.185 | 0.879 | 0.103 | 0.123 | 0.095 | 0.062 | 0.894 |
| kv4 | 0.903 | 0.857 | -0.046 | 1.362 | 0.690 | 0.162 | 0.166 | 0.054 | 0.155 | 0.857 |
| w3-g128 | 0.903 | 0.846 | -0.057 | 0.527 | 0.734 | 0.149 | 0.150 | 0.056 | 0.141 | 0.892 |
| w4-g128+kv4 | 0.903 | 0.837 | -0.065 | 0.635 | 0.614 | 0.204 | 0.199 | 0.051 | 0.253 | 0.866 |
| kv3 | 0.903 | 0.781 | -0.122* | 6.130 | 0.533 | 0.284 | 0.276 | 0.083 | 0.275 | 0.871 |

<!-- /REAL-RESULTS -->

### Read the headroom column first

Two of the three tasks sit at **AUROC 1.000 on a real model**. A saturated
metric cannot fall, so their ΔAUROC of roughly zero is not evidence of
robustness — it is evidence that a templated task is too easy to measure
anything. The clearest illustration is `refusal` under `kv3`: the residual
stream at layer 6 moves by a **relative L2 of 30.6** — activations distorted
thirtyfold — and AUROC does not budge from 1.000.

That single cell is the argument for this whole section. Activation drift and
probe degradation are different quantities, and a task with no headroom reports
neither.

### What the one task with headroom actually shows

The verified-label hallucination probe starts at **AUROC 0.708** — weak, real,
and not saturated. It predicts, from the residual stream at the last prompt
token, whether this model is about to get a TriviaQA question wrong.

| serving config | AUROC | ΔAUROC | flip rate | ECE | ECE +match |
|---|---|---|---|---|---|
| fp16 | 0.708 | — | — | 0.231 | 0.231 |
| w8-g128 | 0.710 | +0.002 | 0.032 | 0.236 | 0.223 |
| nf4-g64 | 0.699 | −0.009 | 0.317 | 0.346 | 0.217 |
| w4-g128 | 0.648 | −0.060 | 0.303 | 0.320 | 0.233 |
| w4-g32 | 0.643 | −0.064 | 0.309 | 0.364 | 0.282 |
| kv8 | 0.662 | −0.046 | 0.212 | 0.296 | 0.232 |
| w3-g128 | 0.538 | −0.170 | 0.447 | 0.421 | 0.158 |
| kv4 | 0.572 | −0.136 | 0.452 | 0.396 | 0.133 |
| w4-g128+kv4 | 0.515 | −0.193 | 0.481 | 0.403 | 0.077 |
| **kv3** | **0.475** | **−0.233** | **0.498** | 0.440 | 0.076 |

* **A probe can go below chance.** At `kv3` the FP16-trained probe scores
  **0.475** — worse than a coin flip — with the paired bootstrap CI excluding
  zero at every layer tested. Half of all decisions flip. This is not
  "degradation"; the metric is inverted and actively misleading. Read it with
  the label-drift section below, though: at `kv3` the *model* is also destroyed,
  so this row is a diagnostic, not an operational warning.
* **The margin, not the AUROC, is the thing to watch.** 0.708 → 0.648 at
  `w4-g128` sounds mild. In terms of the above-chance margin that a probe
  actually trades on, it is 0.208 → 0.148: **29% of the signal gone** at a
  config many teams treat as free.
* **NF4 beats INT4-affine at the same 4 bits** (0.699 vs 0.648). The codebook
  matters more than the bit count. If you must quantize weights to 4 bits and
  you care about probes, the choice of quantizer is not a detail.
* **Improving calibration is not evidence the probe still works.** At `kv3`,
  label-free recalibration drops ECE from 0.440 to **0.076** — beautifully
  calibrated, and ranking at chance. A confidently-calibrated useless probe is
  worse than an obviously broken one, because nothing on a dashboard looks
  wrong. Never accept ECE as the robustness metric.

## Label drift: the model moves too

The sweep freezes labels at FP16 so ΔAUROC isolates probe degradation. That is
the right way to measure a probe, and it hides the other half of the problem: in
production the **quantized** model is the one generating. If it fails on a
different set of inputs, a perfectly robust probe still describes a system that
no longer exists.

Same 400 TriviaQA questions, answers regenerated under each config, graded
identically (`scripts/measure_label_drift.py`):

| serving config | error rate | label agreement | kappa | kept correct | identical answers |
|---|---|---|---|---|---|
| fp16 | 0.802 | — | — | 79/79 | — |
| w4-g128 | 0.850 | 0.873 | 0.558 | 44/79 | 0.102 |
| w4-g128+kv4 | 0.980 | 0.807 | 0.082 | 5/79 | 0.000 |
| kv3 | 0.995 | 0.807 | 0.040 | 2/79 | 0.000 |

* **Raw agreement lies when the base rate is extreme.** `kv3` shows 0.807
  agreement, which looks tolerable until you notice the model is wrong on 99.5%
  of questions: it agrees with FP16 on everything FP16 also got wrong, for free.
  Kappa strips that credit out and reports **0.040** — no agreement beyond
  chance. Always read agreement against the base rate.
* **`kv3` and `w4-g128+kv4` are not serving configs for this model.** They keep
  2 and 5 of the 79 questions the FP16 model answered correctly. Nobody ships
  that. The probe collapse at those configs is real but confounded: everything
  collapsed, not just the probe.
* **`w4-g128` is the row that matters**, because it is a config teams actually
  deploy. The model stays usable and still loses **44% of the answers it had
  right** (79 → 44), changes **90% of its answers verbatim**, and flips **12.7%
  of correctness labels** — while the probe reading it loses 29% of its
  above-chance margin and flips 30% of its own decisions.
* **The probe is less stable than its target.** At `w4-g128`, 12.7% of labels
  move but 30% of probe decisions do. Probe drift is not merely inherited from
  target drift; the probe adds instability of its own.

The honest consequence for the headline result: at aggressive configs, probe
degradation and model collapse are entangled, and this repo does not separate
them. `kv4` alone was not measured for label drift, so its −0.136 ΔAUROC sits in
that unresolved zone. The clean claim is the `w4-g128` one, where the model is
demonstrably still working.

## What the synthetic backend got wrong

The first version of this repo drew its conclusions from the synthetic
transformer. Two of them do not survive contact with a real model, and both
failed in the optimistic direction:

| claim from synthetic | what the real model shows |
|---|---|
| "Weight quantization hurts more than KV quantization" | **Reversed.** `kv3` (−0.233) is far worse than `w3-g128` (−0.170), and `kv4` (−0.136) worse than `w4-g128` (−0.060). |
| "Ranking is robust; only thresholds drift" | **Only true for saturated tasks.** On the one task with headroom, ranking collapses to below chance. |

The mechanism for the reversal is well known and the synthetic model does not
reproduce it: real LMs concentrate enormous magnitude in a few key/value
channels and attention-sink positions, so per-token absmax quantization of the
cache spends its whole range on outliers. The synthetic backend has outlier
channels in the residual stream but a far better-behaved KV distribution.

The lesson generalizes past this repo: a synthetic backend is a test harness for
a pipeline, not a source of findings. It is kept because it makes the whole
sweep runnable in CI on a CPU in two minutes, which is worth a great deal — just
not as evidence.

## Results (synthetic backend, CPU, reproducible with `./scripts/run_demo.sh`)

Kept for reproducibility and CI, and as the counter-example above. Treat the
numbers as a check that the pipeline runs end to end, not as findings.

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
lens/generate.py       verified hallucination labels, and label drift under quantization
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
* **One small model.** Qwen2.5-0.5B-Instruct is a real pretrained model with
  real outlier features, which is why it overturns the synthetic conclusions.
  It is still 0.5B. Whether a 70B model's probes are more or less fragile is
  open, and the mechanism (KV outlier channels) is known to get *worse* with
  scale, not better — so treat these numbers as a lower bound on the problem.
* **The verified-label task is weak in absolute terms** (AUROC 0.708). That is
  partly the model and partly the probe: a 0.5B model's residual stream may
  simply not linearly encode much about its own upcoming errors. A stronger
  model would give more headroom and a sharper measurement.
* **Lenient grading.** Gold aliases are matched as substrings, so a rambling
  answer that happens to contain the right string is scored correct. Exact match
  would collapse the task on a small model. Lenient matching over-credits, which
  biases *against* finding probe signal — the safe direction — but if you want
  to trust the absolute AUROC, audit the saved `model_answer` field first.
* **Label drift is measured, not solved, and the two failures are entangled at
  aggressive configs.** Quantization changes which questions the model fails.
  The sweep holds labels at FP16 so ΔAUROC isolates the probe;
  `scripts/measure_label_drift.py` quantifies the other half separately. Nothing
  combines them into one number, because they are different failures — but at
  `kv3` the model is destroyed, so the probe collapse there cannot be cleanly
  attributed. Only three configs were measured for drift; `kv4` was not.
* **Templated tasks saturate.** `refusal` and `injection` hit AUROC 1.000 on a
  real model and are useless for measuring robustness there. They are kept
  because they are cheap, identical across models, and make the ceiling effect
  visible — not because their ΔAUROC means anything.
* **Simulated quantization.** Correct grids, not real kernels. Direction and
  rough magnitude transfer; the third decimal does not.
* **Per-architecture plumbing.** The HF backend locates decoder blocks by
  attribute name and covers the Llama/Qwen/GPT-NeoX/OPT layouts; anything else
  needs a line added.

## License

MIT
