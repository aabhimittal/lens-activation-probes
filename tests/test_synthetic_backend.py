import numpy as np
import pytest

from lens.backends.synthetic import SyntheticBackend, hash_tokenize
from lens.metrics import activation_drift
from lens.quant import SPECS_BY_NAME

PROMPTS = ["the capital of france is paris", "ignore previous instructions",
           "who discovered the zorblatt engine", "summarize the attached report"]


@pytest.fixture(scope="module")
def be():
    return SyntheticBackend(n_layers=4, d_model=64, n_heads=4, vocab=256, d_ff=128, seed=0)


def test_tokenizer_is_deterministic_and_in_range(be):
    a, b = hash_tokenize("hello world", 256), hash_tokenize("hello world", 256)
    assert a == b and all(0 <= t < 256 for t in a)
    assert hash_tokenize("", 256) == [0]


def test_shapes_and_determinism(be):
    fp = SPECS_BY_NAME["fp16"]
    a = be.activations(PROMPTS, [0, 3], fp)
    b = be.activations(PROMPTS, [0, 3], fp)
    assert a[3].shape == (len(PROMPTS), be.d_model)
    assert np.allclose(a[3], b[3]) and np.isfinite(a[3]).all()


def test_pooling_modes_differ(be):
    fp = SPECS_BY_NAME["fp16"]
    last = be.activations(PROMPTS, [2], fp, "last")[2]
    mean = be.activations(PROMPTS, [2], fp, "mean")[2]
    assert not np.allclose(last, mean)


def test_quantization_actually_perturbs_activations(be):
    fp = be.activations(PROMPTS, [3], SPECS_BY_NAME["fp16"])[3]
    for name in ("w4-g128", "kv4", "w3-g128"):
        q = be.activations(PROMPTS, [3], SPECS_BY_NAME[name])[3]
        assert not np.allclose(fp, q), name
        assert np.isfinite(q).all()


def test_more_aggressive_quantization_drifts_further(be):
    fp = be.activations(PROMPTS, [3], SPECS_BY_NAME["fp16"])[3]

    def drift(name):
        return activation_drift(fp, be.activations(PROMPTS, [3], SPECS_BY_NAME[name])[3])["rel_l2"]

    assert drift("w8-g128") < drift("w4-g128") < drift("w3-g128")
    assert drift("kv8") < drift("kv4") < drift("kv3")
    assert drift("w4-g128+kv3") > drift("w4-g128")


def test_kv_quant_only_touches_the_cache_path(be):
    """Layer 0 still differs (KV quant happens inside every block), but weights
    are untouched, so the weight backup path must be a no-op."""
    spec = SPECS_BY_NAME["kv4"]
    before = be.blocks[0]["wq"].copy()
    be.activations(PROMPTS, [0], spec)
    assert np.array_equal(before, be.blocks[0]["wq"])
