import numpy as np
import pytest

from lens.quant import NF4_CODES, SPECS_BY_NAME, QuantSpec, fake_quant, get_specs

RNG = np.random.default_rng(0)
X = RNG.normal(size=(32, 257)).astype(np.float32)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_passthrough_when_not_quantized():
    assert fake_quant(X, None) is X
    assert fake_quant(X, 16) is X


@pytest.mark.parametrize("scheme", ["affine", "symmetric"])
def test_error_decreases_with_more_bits(scheme):
    errs = [rel(fake_quant(X, b, 128, scheme), X) for b in (3, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


@pytest.mark.parametrize("scheme", ["affine", "symmetric"])
def test_error_decreases_with_smaller_groups(scheme):
    big = rel(fake_quant(X, 4, -1, scheme), X)
    small = rel(fake_quant(X, 4, 32, scheme), X)
    assert small < big


def test_padding_does_not_shift_real_values():
    # 257 is deliberately not a multiple of 32: the edge-repeat padding must not
    # change the scale of the genuine elements in the final group.
    a = fake_quant(X, 8, 32, "symmetric")
    b = fake_quant(X[:, :256], 8, 32, "symmetric")
    assert np.allclose(a[:, :256], b, atol=1e-6)


def test_nf4_uses_only_codebook_levels():
    q = fake_quant(X[:1], 4, -1, "nf4")
    scale = np.abs(X[:1]).max()
    levels = np.unique(np.round(q / scale, 5))
    assert len(levels) <= 16
    for v in levels:
        assert min(abs(v - c) for c in NF4_CODES) < 1e-3


def test_symmetric_grid_is_exact_on_grid_points():
    qmax = 2 ** (8 - 1) - 1
    x = (np.arange(-qmax, qmax + 1, dtype=np.float32) / qmax)[None, :]
    assert np.allclose(fake_quant(x, 8, -1, "symmetric"), x, atol=1e-6)


def test_spec_routing_is_independent():
    w = QuantSpec(name="w", weight_bits=4)
    kv = QuantSpec(name="kv", kv_bits=4)
    assert rel(w.quant_weight(X), X) > 0 and kv.quant_weight(X) is X
    assert rel(kv.quant_kv(X), X) > 0 and w.quant_kv(X) is X


def test_standard_specs_have_exactly_one_baseline():
    assert sum(s.is_baseline for s in get_specs()) == 1
    assert SPECS_BY_NAME["fp16"].is_baseline
    with pytest.raises(KeyError):
        get_specs(["nope"])


def test_torch_and_numpy_agree():
    torch = pytest.importorskip("torch")
    t = torch.from_numpy(X)
    for bits, g, s in [(8, 128, "affine"), (4, 32, "symmetric"), (4, 64, "nf4")]:
        a = fake_quant(X, bits, g, s)
        b = fake_quant(t, bits, g, s).numpy()
        assert np.allclose(a, b, atol=1e-5), (bits, g, s)
