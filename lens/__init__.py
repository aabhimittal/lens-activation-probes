"""LENS -- activation probes as serving-time metrics, under quantization.

Question this repo exists to answer: a linear probe trained on FP16 activations
is cheap enough to run on every token in production. Does it still mean the same
thing when the model underneath is served with INT4 weights or a 3-bit KV cache?
"""
from .experiment import SweepConfig, run_sweep
from .metrics import activation_drift, auroc, score_drift
from .probes import LogisticProbe, MeanDiffProbe, make_probe
from .quant import STANDARD_SPECS, QuantSpec, fake_quant
from .serving import ProbeBundle

__version__ = "0.1.0"
__all__ = ["SweepConfig", "run_sweep", "auroc", "activation_drift", "score_drift",
           "LogisticProbe", "MeanDiffProbe", "make_probe", "QuantSpec", "fake_quant",
           "STANDARD_SPECS", "ProbeBundle"]
