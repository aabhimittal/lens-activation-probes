"""lens command line: sweep, report, export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import SweepConfig, load_results, run_sweep
from .report import build_report


def _cfg_from_args(a) -> SweepConfig:
    if a.config:
        cfg = SweepConfig.from_yaml(a.config)
    else:
        cfg = SweepConfig()
    for k in ("backend", "n_examples", "pooling", "seed", "calib_n"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    if a.model:
        cfg.backend = "hf"
        cfg.backend_kwargs = {**cfg.backend_kwargs, "model_id": a.model}
    if a.tasks:
        cfg.tasks = a.tasks
    if a.layers:
        cfg.layers = a.layers
    if a.specs:
        cfg.specs = a.specs
    if a.probes:
        cfg.probes = a.probes
    return cfg


def cmd_sweep(a) -> int:
    cfg = _cfg_from_args(a)
    rows = run_sweep(cfg, out_dir=a.out)
    md = build_report(rows, title=f"LENS sweep ({cfg.backend})")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "report.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {a.out}/results.jsonl, {a.out}/report.md")
    return 0


def cmd_report(a) -> int:
    rows = [r for f in a.results for r in load_results(f)]
    md = build_report(rows, title=a.title)
    if a.drift:
        from .report import entanglement_table
        drift = json.loads(Path(a.drift).read_text())["rows"]
        md += ("\n\n## Probe degradation vs model collapse\n\n"
               + entanglement_table(rows, drift, task=a.drift_task,
                                    health_floor=a.health_floor) + "\n")
    if a.out:
        Path(a.out).write_text(md + "\n")
    print(md)
    return 0


def cmd_export(a) -> int:
    """Train one probe on FP16 activations and write a serving bundle, including
    per-config affine corrections so the same weights serve a quantized stack."""
    from .backends import get_backend
    from .data import get_dataset, split
    from .probes import make_probe, recalibrate_match
    from .quant import get_specs
    from .serving import bundle_from_probe

    cfg = _cfg_from_args(a)
    backend = get_backend(cfg.backend, **cfg.backend_kwargs)
    layer = (cfg.layers or [backend.n_layers // 2])[0]
    ds = get_dataset(cfg.tasks[0], n=cfg.n_examples, seed=cfg.seed)
    tr, _ = split(ds, cfg.test_frac, cfg.seed, cfg.split_by_template)
    specs = get_specs(cfg.specs)
    base = next(s for s in specs if s.is_baseline)
    acts = {s.name: backend.activations(tr.texts, [layer], s, cfg.pooling)[layer] for s in specs}
    probe = make_probe(cfg.probes[0]).fit(acts[base.name], tr.labels)
    corrections = {}
    for s in specs:
        if s.is_baseline:
            continue
        m = recalibrate_match(probe, acts[base.name][: cfg.calib_n], acts[s.name][: cfg.calib_n])
        a_ = float(np.dot(m.w, probe.w) / (np.dot(probe.w, probe.w) + 1e-12))
        corrections[s.name] = [a_, float(m.b - a_ * probe.b)]
    bundle = bundle_from_probe(probe, layer, cfg.tasks[0], pooling=cfg.pooling,
                              model=backend.name, corrections=corrections,
                              metadata={"n_train": len(tr), "probe": cfg.probes[0]})
    p = bundle.save(a.bundle)
    print(f"wrote {p.with_suffix('.npz')} and {p.with_suffix('.json')}")
    print(json.dumps({"layer": layer, "d": int(probe.w.size),
                      "corrections": corrections}, indent=2))
    return 0


def cmd_label(a) -> int:
    """Generate answers with the FP16 reference stack, grade them, and write a
    verified-label dataset the sweep can consume with --tasks <path>."""
    from .backends import get_backend
    from .generate import build_verified_hallucination, save_labeled

    be = get_backend("hf", model_id=a.model, device=a.device, dtype=a.dtype,
                     batch_size=a.batch_size, max_len=a.max_len)
    ds, stats = build_verified_hallucination(be, n=a.n, seed=a.seed or 0,
                                             max_new_tokens=a.max_new_tokens,
                                             balance=not a.no_balance,
                                             prompt_style=a.prompt_style)
    p = save_labeled(ds, stats, a.out)
    print(json.dumps(stats, indent=2))
    print(f"wrote {p} ({len(ds)} examples) and {p.with_suffix('.meta.json')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("lens", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config")
        p.add_argument("--backend")
        p.add_argument("--model", help="HF model id (implies --backend hf)")
        p.add_argument("--tasks", nargs="+")
        p.add_argument("--layers", nargs="+", type=int)
        p.add_argument("--specs", nargs="+")
        p.add_argument("--probes", nargs="+")
        p.add_argument("--n-examples", dest="n_examples", type=int)
        p.add_argument("--pooling")
        p.add_argument("--calib-n", dest="calib_n", type=int)
        p.add_argument("--seed", type=int)

    s = sub.add_parser("sweep", help="run the quantization sweep")
    common(s)
    s.add_argument("--out", default="results/latest")
    s.set_defaults(fn=cmd_sweep)

    r = sub.add_parser("report", help="rebuild markdown from results.jsonl")
    r.add_argument("results", nargs="+", help="one or more results.jsonl to merge")
    r.add_argument("--out")
    r.add_argument("--title", default="LENS sweep")
    r.add_argument("--drift", help="label-drift JSON to join against, per config")
    r.add_argument("--drift-task", dest="drift_task",
                   help="substring of the task the drift labels belong to")
    r.add_argument("--health-floor", dest="health_floor", type=float, default=0.8,
                   help="fraction of FP16-correct answers a config must retain "
                        "to count toward the confound-free average")
    r.set_defaults(fn=cmd_report)

    e = sub.add_parser("export", help="train one probe and write a serving bundle")
    common(e)
    e.add_argument("--bundle", default="artifacts/probe")
    e.set_defaults(fn=cmd_export)

    l = sub.add_parser("label", help="build verified hallucination labels for a model")
    l.add_argument("--model", required=True)
    l.add_argument("--out", default="data/hallucination_verified.jsonl")
    l.add_argument("--n", type=int, default=800, help="questions to sample before balancing")
    l.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=16)
    l.add_argument("--prompt-style", dest="prompt_style", default="auto",
                   choices=["auto", "chat", "plain"])
    l.add_argument("--device", default="auto")
    l.add_argument("--dtype", default="auto")
    l.add_argument("--max-len", dest="max_len", type=int, default=512)
    l.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    l.add_argument("--no-balance", action="store_true")
    l.add_argument("--seed", type=int, default=0)
    l.set_defaults(fn=cmd_label)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
