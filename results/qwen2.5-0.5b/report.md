# LENS sweep (hf)

240 rows | tasks: data/hallucination_qwen2.5-0.5b.jsonl, injection, refusal | layers: [6, 12, 18, 21] | probes: logistic, meandiff

## Probe families under quantization

| probe | AUROC fp16 | AUROC quantized | ΔAUROC | flip rate | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|
| logistic | 0.903 | 0.864 | -0.039 | 0.132 | 0.119 | 0.889 |
| meandiff | 0.895 | 0.860 | -0.035 | 0.124 | 0.119 | 0.881 |

## logistic probe: degradation by serving config

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

### logistic: ΔAUROC per task

| serving config | Δ data/hallucination_qwen2.5-0.5b.jsonl | Δ injection | Δ refusal |
|---|---|---|---|
| fp16 | 0.000 | 0.000 | 0.000 |
| kv3 | -0.233 | -0.005 | -0.127 |
| kv4 | -0.136 | -0.002 | 0.000 |
| kv8 | -0.046 | 0.000 | 0.000 |
| nf4-g64 | -0.009 | 0.000 | 0.000 |
| w3-g128 | -0.170 | 0.000 | 0.000 |
| w4-g128 | -0.060 | 0.000 | 0.000 |
| w4-g128+kv4 | -0.193 | -0.000 | -0.003 |
| w4-g32 | -0.064 | 0.000 | 0.000 |
| w8-g128 | 0.002 | 0.000 | 0.000 |

## meandiff probe: degradation by serving config

| serving config | AUROC fp16 | AUROC | ΔAUROC | act rel-L2 | score r | flip rate | ECE | ECE +match | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 0.895 | 0.895 | 0.000 | 0.000 | 1.000 | 0.000 | 0.089 | 0.089 | 0.000 | 0.895 |
| w8-g128 | 0.895 | 0.895 | -0.000 | 0.015 | 0.998 | 0.014 | 0.083 | 0.085 | 0.004 | 0.895 |
| nf4-g64 | 0.895 | 0.889 | -0.006 | 0.202 | 0.872 | 0.126 | 0.111 | 0.083 | 0.063 | 0.892 |
| kv8 | 0.895 | 0.884 | -0.011 | 0.135 | 0.911 | 0.065 | 0.089 | 0.075 | 0.053 | 0.895 |
| w4-g128 | 0.895 | 0.872 | -0.023 | 0.234 | 0.864 | 0.113 | 0.117 | 0.075 | 0.082 | 0.897 |
| w4-g32 | 0.895 | 0.871 | -0.024 | 0.185 | 0.867 | 0.122 | 0.122 | 0.079 | 0.071 | 0.899 |
| kv4 | 0.895 | 0.851 | -0.044 | 1.362 | 0.696 | 0.147 | 0.154 | 0.056 | 0.151 | 0.847 |
| w4-g128+kv4 | 0.895 | 0.842 | -0.053 | 0.635 | 0.666 | 0.155 | 0.159 | 0.113 | 0.248 | 0.860 |
| w3-g128 | 0.895 | 0.837 | -0.058 | 0.527 | 0.725 | 0.163 | 0.164 | 0.046 | 0.145 | 0.881 |
| kv3 | 0.895 | 0.795 | -0.100* | 6.130 | 0.600 | 0.215 | 0.213 | 0.127 | 0.250 | 0.863 |

### meandiff: ΔAUROC per task

| serving config | Δ data/hallucination_qwen2.5-0.5b.jsonl | Δ injection | Δ refusal |
|---|---|---|---|
| fp16 | 0.000 | 0.000 | 0.000 |
| kv3 | -0.193 | 0.000 | -0.106 |
| kv4 | -0.125 | -0.006 | 0.000 |
| kv8 | -0.034 | 0.000 | 0.000 |
| nf4-g64 | -0.018 | 0.000 | 0.000 |
| w3-g128 | -0.175 | 0.000 | 0.000 |
| w4-g128 | -0.068 | 0.000 | 0.000 |
| w4-g128+kv4 | -0.159 | 0.000 | 0.000 |
| w4-g32 | -0.071 | 0.000 | 0.000 |
| w8-g128 | -0.001 | 0.000 | 0.000 |

## Layer sensitivity (mean over quantized configs)

| layer | AUROC fp16 | mean ΔAUROC (quantized) | act rel-L2 | flip rate |
|---|---|---|---|---|
| 6 | 0.915 | -0.052 | 1.647 | 0.157 |
| 12 | 0.899 | -0.046 | 1.370 | 0.145 |
| 18 | 0.896 | -0.032 | 0.843 | 0.127 |
| 21 | 0.900 | -0.026 | 0.329 | 0.100 |

## Headline numbers

- Mean ΔAUROC across all quantized configs: **-0.037** (mean flip rate at the deployed threshold: **0.128**).
- Worst single cell: `kv3` on `data/hallucination_qwen2.5-0.5b.jsonl` layer 6 (logistic), ΔAUROC **-0.296**.
- Weight-only INT4 configs: mean ΔAUROC -0.034, mean flip rate 0.133.
- KV-cache-only configs: mean ΔAUROC -0.056, mean flip rate 0.157.
- 46/216 quantized cells show a ΔAUROC whose paired 95% bootstrap CI excludes zero; the rest are inside the noise floor.
- Repairs: label-free affine matching cuts mean flip rate 0.128 -> 0.119; retraining on quantized activations moves mean AUROC 0.862 -> 0.885.

Columns: ΔAUROC is quantized minus FP16 for the *unchanged* FP16-trained probe, marked `*` when its paired bootstrap CI excludes zero. `flip rate` is the fraction of test decisions that change at the deployed threshold -- the operational cost AUROC cannot see. `+match` is label-free affine recalibration on paired activations; `+refit` retrains on quantized activations -- the strongest linear repair available, though not guaranteed to help: a direction fit on degraded activations can be worse than one transferred from clean ones.
