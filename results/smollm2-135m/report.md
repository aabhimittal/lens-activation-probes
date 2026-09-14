# LENS sweep (hf)

160 rows | tasks: injection, refusal | layers: [7, 15, 22, 26] | probes: logistic, meandiff

## Probe families under quantization

| probe | AUROC fp16 | AUROC quantized | ΔAUROC | flip rate | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|
| logistic | 1.000 | 1.000 | -0.000 | 0.001 | 0.000 | 1.000 |
| meandiff | 1.000 | 1.000 | -0.000 | 0.000 | 0.007 | 1.000 |

## logistic probe: degradation by serving config

| serving config | AUROC fp16 | AUROC | ΔAUROC | act rel-L2 | score r | flip rate | ECE | ECE +match | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.002 | 0.002 | 0.000 | 1.000 |
| w8-g128 | 1.000 | 1.000 | 0.000 | 0.014 | 1.000 | 0.000 | 0.002 | 0.002 | 0.000 | 1.000 |
| w4-g128 | 1.000 | 1.000 | 0.000 | 0.200 | 1.000 | 0.000 | 0.005 | 0.003 | 0.000 | 1.000 |
| w4-g32 | 1.000 | 1.000 | 0.000 | 0.165 | 1.000 | 0.000 | 0.003 | 0.002 | 0.000 | 1.000 |
| nf4-g64 | 1.000 | 1.000 | 0.000 | 0.168 | 1.000 | 0.000 | 0.004 | 0.002 | 0.000 | 1.000 |
| kv8 | 1.000 | 1.000 | 0.000 | 0.008 | 1.000 | 0.000 | 0.002 | 0.002 | 0.000 | 1.000 |
| kv4 | 1.000 | 1.000 | 0.000 | 0.135 | 1.000 | 0.000 | 0.003 | 0.002 | 0.000 | 1.000 |
| kv3 | 1.000 | 1.000 | 0.000 | 0.269 | 1.000 | 0.000 | 0.009 | 0.005 | 0.000 | 1.000 |
| w4-g128+kv4 | 1.000 | 1.000 | 0.000 | 0.244 | 1.000 | 0.000 | 0.007 | 0.004 | 0.000 | 1.000 |
| w3-g128 | 1.000 | 0.999 | -0.001 | 0.412 | 0.985 | 0.005 | 0.041 | 0.091 | 0.001 | 1.000 |

### logistic: ΔAUROC per task

| serving config | Δ injection | Δ refusal |
|---|---|---|
| fp16 | 0.000 | 0.000 |
| kv3 | 0.000 | 0.000 |
| kv4 | 0.000 | 0.000 |
| kv8 | 0.000 | 0.000 |
| nf4-g64 | 0.000 | 0.000 |
| w3-g128 | 0.000 | -0.001 |
| w4-g128 | 0.000 | 0.000 |
| w4-g128+kv4 | 0.000 | 0.000 |
| w4-g32 | 0.000 | 0.000 |
| w8-g128 | 0.000 | 0.000 |

## meandiff probe: degradation by serving config

| serving config | AUROC fp16 | AUROC | ΔAUROC | act rel-L2 | score r | flip rate | ECE | ECE +match | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| w8-g128 | 1.000 | 1.000 | 0.000 | 0.014 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| w4-g128 | 1.000 | 1.000 | 0.000 | 0.200 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| w4-g32 | 1.000 | 1.000 | 0.000 | 0.165 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| nf4-g64 | 1.000 | 1.000 | 0.000 | 0.168 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| kv8 | 1.000 | 1.000 | 0.000 | 0.008 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| kv4 | 1.000 | 1.000 | 0.000 | 0.135 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| kv3 | 1.000 | 1.000 | 0.000 | 0.269 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| w4-g128+kv4 | 1.000 | 1.000 | 0.000 | 0.244 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| w3-g128 | 1.000 | 1.000 | -0.000 | 0.412 | 0.998 | 0.001 | 0.001 | 0.107 | 0.062 | 1.000 |

### meandiff: ΔAUROC per task

| serving config | Δ injection | Δ refusal |
|---|---|---|
| fp16 | 0.000 | 0.000 |
| kv3 | 0.000 | 0.000 |
| kv4 | 0.000 | 0.000 |
| kv8 | 0.000 | 0.000 |
| nf4-g64 | 0.000 | 0.000 |
| w3-g128 | 0.000 | -0.001 |
| w4-g128 | 0.000 | 0.000 |
| w4-g128+kv4 | 0.000 | 0.000 |
| w4-g32 | 0.000 | 0.000 |
| w8-g128 | 0.000 | 0.000 |

## Layer sensitivity (mean over quantized configs)

| layer | AUROC fp16 | mean ΔAUROC (quantized) | act rel-L2 | flip rate |
|---|---|---|---|---|
| 7 | 1.000 | 0.000 | 0.158 | 0.000 |
| 15 | 1.000 | 0.000 | 0.166 | 0.000 |
| 22 | 1.000 | 0.000 | 0.194 | 0.001 |
| 26 | 1.000 | -0.000 | 0.199 | 0.002 |

## Headline numbers

- Mean ΔAUROC across all quantized configs: **-0.000** (mean flip rate at the deployed threshold: **0.000**).
- Worst single cell: `w3-g128` on `refusal` layer 26 (logistic), ΔAUROC **-0.004**.
- Weight-only INT4 configs: mean ΔAUROC 0.000, mean flip rate 0.000.
- KV-cache-only configs: mean ΔAUROC 0.000, mean flip rate 0.000.
- 0/144 quantized cells show a ΔAUROC whose paired 95% bootstrap CI excludes zero; the rest are inside the noise floor.
- Repairs: label-free affine matching cuts mean flip rate 0.000 -> 0.004; retraining on quantized activations moves mean AUROC 1.000 -> 1.000.

Columns: ΔAUROC is quantized minus FP16 for the *unchanged* FP16-trained probe, marked `*` when its paired bootstrap CI excludes zero. `flip rate` is the fraction of test decisions that change at the deployed threshold -- the operational cost AUROC cannot see. `+match` is label-free affine recalibration on paired activations; `+refit` retrains on quantized activations -- the strongest linear repair available, though not guaranteed to help: a direction fit on degraded activations can be worse than one transferred from clean ones.
