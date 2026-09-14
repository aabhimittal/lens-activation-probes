import numpy as np

from lens.data import get_dataset, split
from lens.experiment import SweepConfig, load_results, run_sweep
from lens.probes import LogisticProbe
from lens.report import build_report
from lens.serving import ProbeBundle, bundle_from_probe

SMALL = dict(n_layers=3, d_model=48, n_heads=4, vocab=256, d_ff=96, seed=0)


def cfg(**kw):
    base = dict(backend="synthetic", backend_kwargs=SMALL, tasks=["injection"],
                layers=[1, 2], probes=["logistic"], specs=["fp16", "w4-g128", "kv3"],
                n_examples=120, calib_n=32, bootstrap=25)
    base.update(kw)
    return SweepConfig(**base)


def test_datasets_are_balanced_and_template_disjoint():
    for name in ("refusal", "hallucination", "injection"):
        ds = get_dataset(name, 120)
        tr, te = split(ds, by_template=True)
        assert len(tr) + len(te) == len(ds)
        assert not (set(tr.groups) & set(te.groups))
        assert 0.3 < ds.labels.mean() < 0.7


def test_sweep_shape_and_baseline_invariants(tmp_path):
    rows = run_sweep(cfg(), out_dir=tmp_path, verbose=False)
    assert len(rows) == 1 * 2 * 1 * 3  # tasks x layers x probes x specs
    base = [r for r in rows if r["spec"] == "fp16"]
    assert len(base) == 2
    for r in base:
        # The baseline must be numerically identical to itself, or the
        # extraction path is non-deterministic and every delta is noise.
        assert r["delta_auroc"] == 0.0
        assert r["score_flip_rate"] == 0.0
        assert r["act_rel_l2"] < 1e-9
    for r in rows:
        if r["spec"] != "fp16":
            assert r["act_rel_l2"] > 0
        assert 0.0 <= r["auroc"] <= 1.0
    assert len(load_results(tmp_path / "results.jsonl")) == len(rows)
    assert (tmp_path / "config.json").exists()


def test_report_renders_every_section(tmp_path):
    md = build_report(run_sweep(cfg(), verbose=False), title="t")
    for frag in ("# t", "serving config", "Layer sensitivity", "Headline numbers",
                 "flip +match"):
        assert frag in md


def test_serving_bundle_roundtrip_and_correction(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    X = rng.normal(size=(300, 16)) + np.outer(y, rng.normal(size=16)) * 0.5
    p = LogisticProbe().fit(X, y)
    b = bundle_from_probe(p, layer=2, task="injection", corrections={"kv3": [0.5, 0.1]})
    f = b.save(tmp_path / "probe")
    b2 = ProbeBundle.load(f)
    assert np.allclose(b2.score(X), p.predict_proba(X))
    assert b2.layer == 2 and b2.task == "injection"
    assert not np.allclose(b2.score(X, spec="kv3"), b2.score(X))
    assert np.array_equal(b2.flag(X), b2.score(X) >= b2.threshold)


def test_cli_sweep_runs(tmp_path):
    from lens.cli import main
    out = tmp_path / "r"
    assert main(["sweep", "--backend", "synthetic", "--tasks", "injection",
                 "--specs", "fp16", "kv4", "--probes", "meandiff",
                 "--layers", "1", "--n-examples", "100", "--out", str(out)]) == 0
    assert (out / "report.md").exists() and (out / "results.jsonl").exists()
