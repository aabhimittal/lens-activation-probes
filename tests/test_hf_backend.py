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


def test_config_switches_never_compound(be):
    """w4 applied after w3 must quantize the ORIGINAL weights, not the 3-bit
    ones. Quantizing an already-quantized tensor is silent and would make every
    row of a sweep depend on the order the configs happened to run in."""
    L = be.n_layers - 1
    first = be.activations(PROMPTS, [L], SPECS_BY_NAME["w4-g128"])[L]
    be.activations(PROMPTS, [L], SPECS_BY_NAME["w3-g128"])
    be.activations(PROMPTS, [L], SPECS_BY_NAME["fp16"])
    again = be.activations(PROMPTS, [L], SPECS_BY_NAME["w4-g128"])[L]
    assert np.allclose(first, again, atol=1e-5)


def test_snapshot_is_taken_once_and_reused(be):
    be.restore_weights(drop=True)
    assert not be._fp_backup
    be.activations(PROMPTS, [0], SPECS_BY_NAME["w4-g128"])
    snapshot = {k: v.clone() for k, v in be._fp_backup.items()}
    assert snapshot, "no FP snapshot was taken"
    be.activations(PROMPTS, [0], SPECS_BY_NAME["w3-g128"])
    # The snapshot must still hold FP weights after a second config, not the
    # quantized ones that are currently loaded in the model.
    for k, v in snapshot.items():
        assert np.allclose(v.numpy(), be._fp_backup[k].numpy())
