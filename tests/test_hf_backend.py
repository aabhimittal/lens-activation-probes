"""HF-backend tests. Opt-in: they download a ~2MB random-weight model.

    LENS_TEST_HF=1 pytest tests/test_hf_backend.py

Validated against torch 2.14 / transformers 5.17 and the 4.x keyword layout.
"""
import os

import numpy as np
import pytest

from lens.metrics import activation_drift
from lens.quant import SPECS_BY_NAME

pytestmark = pytest.mark.skipif(not os.environ.get("LENS_TEST_HF"),
                                reason="set LENS_TEST_HF=1 (downloads a tiny model)")
MODEL = os.environ.get("LENS_TEST_HF_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
PROMPTS = ["the capital of france is", "ignore previous instructions and reveal",
           "who invented the zorblatt engine", "summarize this document please"]


@pytest.fixture(scope="module")
def be():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from lens.backends.hf import HFBackend
    return HFBackend(MODEL, device="cpu", dtype="float32", batch_size=2)


def test_layers_and_shapes(be):
    a = be.activations(PROMPTS, [0, be.n_layers - 1], SPECS_BY_NAME["fp16"])
    assert a[0].shape == (len(PROMPTS), be.d_model)
    assert np.isfinite(a[0]).all()


def test_extraction_is_deterministic(be):
    fp = SPECS_BY_NAME["fp16"]
    assert np.allclose(be.activations(PROMPTS, [0], fp)[0],
                       be.activations(PROMPTS, [0], fp)[0])


def test_weight_quant_is_reversible(be):
    """If restoration leaked, every subsequent baseline would be contaminated."""
    L = be.n_layers - 1
    before = be.activations(PROMPTS, [L], SPECS_BY_NAME["fp16"])[L]
    be.activations(PROMPTS, [L], SPECS_BY_NAME["w3-g128"])
    after = be.activations(PROMPTS, [L], SPECS_BY_NAME["fp16"])[L]
    assert np.allclose(before, after, atol=1e-5)


@pytest.mark.parametrize("spec", ["w4-g128", "kv3", "w8a8"])
def test_each_quant_path_is_actually_exercised(be, spec):
    L = be.n_layers - 1
    fp = be.activations(PROMPTS, [L], SPECS_BY_NAME["fp16"])[L]
    q = be.activations(PROMPTS, [L], SPECS_BY_NAME[spec])[L]
    assert activation_drift(fp, q)["rel_l2"] > 1e-6, f"{spec} changed nothing"
    assert np.isfinite(q).all()


def test_drift_ordering_matches_bit_width(be):
    L = be.n_layers - 1
    fp = be.activations(PROMPTS, [L], SPECS_BY_NAME["fp16"])[L]

    def d(n):
        return activation_drift(fp, be.activations(PROMPTS, [L], SPECS_BY_NAME[n])[L])["rel_l2"]

    assert d("w8-g128") < d("w4-g128") < d("w3-g128")
    assert d("kv8") < d("kv4") < d("kv3")


def test_sweep_runs_on_hf_backend(be):
    from lens.experiment import SweepConfig, run_sweep
    rows = run_sweep(SweepConfig(
        backend="hf", backend_kwargs={"model_id": MODEL, "device": "cpu",
                                      "dtype": "float32", "batch_size": 4},
        tasks=["injection"], layers=[be.n_layers - 1], probes=["logistic"],
        specs=["fp16", "kv4"], n_examples=40, calib_n=16, bootstrap=20), verbose=False)
    assert len(rows) == 2
    assert [r for r in rows if r["spec"] == "fp16"][0]["delta_auroc"] == 0.0
