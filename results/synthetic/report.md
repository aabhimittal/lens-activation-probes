# LENS sweep

312 rows | tasks: hallucination, injection, refusal | layers: [2, 4, 5, 6] | probes: logistic, meandiff

## Probe families under quantization

| probe | AUROC fp16 | AUROC quantized | ΔAUROC | flip rate | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|
| logistic | 0.915 | 0.909 | -0.006 | 0.056 | 0.055 | 0.908 |
| meandiff | 0.900 | 0.895 | -0.005 | 0.064 | 0.058 | 0.908 |

## logistic probe: degradation by serving config

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

### logistic: ΔAUROC per task

| serving config | Δ hallucination | Δ injection | Δ refusal |
|---|---|---|---|
| fp16 | 0.000 | 0.000 | 0.000 |
| kv3 | 0.008 | -0.001 | -0.009 |
| kv4 | 0.016 | -0.006 | -0.007 |
| kv8 | 0.000 | 0.000 | 0.000 |
| nf4-g64 | -0.019 | -0.005 | -0.014 |
| w3-g128 | -0.045 | -0.043 | -0.057 |
| w4-g128 | -0.001 | -0.011 | 0.003 |
| w4-g128+kv3 | 0.011 | -0.003 | 0.003 |
| w4-g128+kv4 | -0.009 | -0.018 | -0.001 |
| w4-g128+kv8 | -0.003 | -0.011 | 0.003 |
| w4-g32 | 0.019 | -0.008 | -0.006 |
| w8-g128 | 0.007 | 0.000 | -0.002 |
| w8a8 | 0.000 | -0.000 | -0.003 |

## meandiff probe: degradation by serving config

| serving config | AUROC fp16 | AUROC | ΔAUROC | act rel-L2 | score r | flip rate | ECE | ECE +match | flip +match | AUROC +refit |
|---|---|---|---|---|---|---|---|---|---|---|
| w4-g32 | 0.900 | 0.909 | 0.009 | 0.234 | 0.865 | 0.050 | 0.266 | 0.258 | 0.040 | 0.918 |
| kv4 | 0.900 | 0.909 | 0.009 | 0.114 | 0.947 | 0.026 | 0.249 | 0.238 | 0.036 | 0.917 |
| w4-g128+kv3 | 0.900 | 0.908 | 0.008 | 0.368 | 0.778 | 0.089 | 0.271 | 0.281 | 0.082 | 0.916 |
| kv3 | 0.900 | 0.901 | 0.001 | 0.232 | 0.885 | 0.071 | 0.256 | 0.255 | 0.060 | 0.919 |
| w8-g128 | 0.900 | 0.901 | 0.001 | 0.017 | 0.999 | 0.000 | 0.253 | 0.253 | 0.000 | 0.903 |
| kv8 | 0.900 | 0.901 | 0.001 | 0.007 | 1.000 | 0.000 | 0.252 | 0.253 | 0.000 | 0.901 |
| fp16 | 0.900 | 0.900 | 0.000 | 0.000 | 1.000 | 0.000 | 0.253 | 0.253 | 0.000 | 0.900 |
| w4-g128+kv8 | 0.900 | 0.899 | -0.001 | 0.284 | 0.777 | 0.098 | 0.284 | 0.288 | 0.102 | 0.937 |
| w4-g128 | 0.900 | 0.899 | -0.001 | 0.284 | 0.777 | 0.096 | 0.284 | 0.287 | 0.099 | 0.938 |
| w8a8 | 0.900 | 0.898 | -0.001 | 0.030 | 0.993 | 0.008 | 0.257 | 0.258 | 0.009 | 0.899 |
| w4-g128+kv4 | 0.900 | 0.896 | -0.004 | 0.304 | 0.797 | 0.090 | 0.294 | 0.286 | 0.091 | 0.910 |
| nf4-g64 | 0.900 | 0.887 | -0.013 | 0.262 | 0.763 | 0.051 | 0.273 | 0.260 | 0.041 | 0.897 |
| w3-g128 | 0.900 | 0.834 | -0.066* | 0.566 | 0.688 | 0.186 | 0.254 | 0.287 | 0.140 | 0.836 |

### meandiff: ΔAUROC per task

| serving config | Δ hallucination | Δ injection | Δ refusal |
|---|---|---|---|
| fp16 | 0.000 | 0.000 | 0.000 |
| kv3 | 0.022 | -0.016 | -0.001 |
| kv4 | 0.027 | -0.000 | 0.000 |
| kv8 | 0.003 | 0.000 | 0.000 |
| nf4-g64 | -0.018 | -0.006 | -0.013 |
| w3-g128 | -0.066 | -0.043 | -0.089 |
| w4-g128 | 0.011 | -0.012 | -0.001 |
| w4-g128+kv3 | 0.032 | -0.008 | -0.001 |
| w4-g128+kv4 | 0.005 | -0.017 | 0.000 |
| w4-g128+kv8 | 0.011 | -0.012 | -0.001 |
| w4-g32 | 0.030 | -0.002 | 0.000 |
| w8-g128 | 0.004 | -0.000 | 0.000 |
| w8a8 | -0.004 | -0.000 | 0.000 |

## Layer sensitivity (mean over quantized configs)

| layer | AUROC fp16 | mean ΔAUROC (quantized) | act rel-L2 | flip rate |
|---|---|---|---|---|
| 2 | 0.947 | 0.001 | 0.215 | 0.035 |
| 4 | 0.911 | -0.005 | 0.225 | 0.065 |
| 5 | 0.908 | -0.010 | 0.227 | 0.068 |
| 6 | 0.894 | -0.009 | 0.234 | 0.054 |

## Headline numbers

- Mean ΔAUROC across all quantized configs: **-0.005** (mean flip rate at the deployed threshold: **0.060**).
- Worst single cell: `w3-g128` on `hallucination` layer 2 (meandiff), ΔAUROC **-0.134**.
- Weight-only INT4 configs: mean ΔAUROC 0.000, mean flip rate 0.077.
- KV-cache-only configs: mean ΔAUROC 0.002, mean flip rate 0.031.
- 72/288 quantized cells show a ΔAUROC whose paired 95% bootstrap CI excludes zero; the rest are inside the noise floor.
- Repairs: label-free affine matching cuts mean flip rate 0.060 -> 0.056; retraining on quantized activations moves mean AUROC 0.902 -> 0.908.

Columns: ΔAUROC is quantized minus FP16 for the *unchanged* FP16-trained probe, marked `*` when its paired bootstrap CI excludes zero. `flip rate` is the fraction of test decisions that change at the deployed threshold -- the operational cost AUROC cannot see. `+match` is label-free affine recalibration on paired activations; `+refit` retrains on quantized activations -- the strongest linear repair available, though not guaranteed to help: a direction fit on degraded activations can be worse than one transferred from clean ones.
