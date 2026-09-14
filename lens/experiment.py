"""The sweep: train probes on FP16 activations, then measure what happens when
the serving stack is quantized underneath them.

Protocol (one row per task x layer x probe x quant spec):
  1. Extract FP16 activations for train/test prompts. Train the probe. -> baseline AUROC.
  2. For each quant spec, re-extract activations for the *same* prompts.
  3. Score the test set with the *unchanged* FP16-trained probe. -> transferred AUROC.
  4. Compute repairs:
       match  - label-free affine recalibration fitted on paired FP/quant
                activations from a small calibration slice of the train set;
       affine - 2-parameter recalibration fitted on labeled quantized calibration data;
       refit  - retrain the probe from scratch on quantized activations (upper bound).
  5. Record activation drift and probe-score drift alongside, because AUROC is
     rank-based and therefore blind to the failure that actually bites in
     production: a threshold that silently moves.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .backends import get_backend
from .data import Dataset, get_dataset, split
from .metrics import (activation_drift, auroc, bootstrap_ci, classification_report,
                      paired_delta_ci, score_drift)
from .probes import make_probe, recalibrate_affine, recalibrate_match, refit_on_quantized
from .quant import QuantSpec, get_specs


@dataclass
class SweepConfig:
    backend: str = "synthetic"
    backend_kwargs: dict = field(default_factory=dict)
    tasks: list[str] = field(default_factory=lambda: ["refusal", "hallucination", "injection"])
    layers: Optional[list[int]] = None           # None -> quartile layers
    probes: list[str] = field(default_factory=lambda: ["logistic", "meandiff"])
    specs: Optional[list[str]] = None            # None -> all STANDARD_SPECS
    n_examples: int = 400
    pooling: str = "last"
    test_frac: float = 0.3
    split_by_template: bool = True
    calib_n: int = 64                            # examples used by the repairs
    seed: int = 0
    threshold: float = 0.5
    bootstrap: int = 200

    @classmethod
    def from_yaml(cls, path) -> "SweepConfig":
        import yaml
        return cls(**yaml.safe_load(Path(path).read_text()))


def _default_layers(n_layers: int) -> list[int]:
    """Quartile depths. Mid-to-late layers carry the most linearly decodable
    task structure; the last layer is already specialized for next-token
    prediction and is usually a worse probe site."""
    return sorted({max(0, int(round(f * (n_layers - 1)))) for f in (0.25, 0.5, 0.75, 0.9)})


def run_sweep(cfg: SweepConfig, out_dir: Optional[str] = None, verbose: bool = True) -> list[dict]:
    backend = get_backend(cfg.backend, **cfg.backend_kwargs)
    layers = cfg.layers if cfg.layers is not None else _default_layers(backend.n_layers)
    specs = get_specs(cfg.specs)
    baseline = next((s for s in specs if s.is_baseline), None)
    if baseline is None:
        raise ValueError("sweep needs the fp16 baseline spec in the list")
    rows: list[dict] = []

    for task in cfg.tasks:
        ds = get_dataset(task, n=cfg.n_examples, seed=cfg.seed)
        tr, te = split(ds, cfg.test_frac, cfg.seed, cfg.split_by_template)
        prompts = list(tr.texts) + list(te.texts)
        n_tr = len(tr)
        t0 = time.time()
        acts = {}
        for si, s in enumerate(specs, 1):
            ts = time.time()
            acts[s.name] = backend.activations(prompts, layers, s, cfg.pooling)
            if verbose:
                # Per-spec, not per-task: a real-model sweep spends tens of
                # minutes here and a single line at the end is not observable.
                print(f"[{task}] {si}/{len(specs)} {s.name}: "
                      f"{len(prompts)} prompts in {time.time() - ts:.1f}s", flush=True)
        if verbose:
            print(f"[{task}] {len(prompts)} prompts x {len(specs)} specs "
                  f"in {time.time() - t0:.1f}s", flush=True)

        rng = np.random.default_rng(cfg.seed)
        cal_idx = rng.choice(n_tr, size=min(cfg.calib_n, n_tr), replace=False)

        for layer in layers:
            Xtr_fp = acts[baseline.name][layer][:n_tr]
            Xte_fp = acts[baseline.name][layer][n_tr:]
            for kind in cfg.probes:
                probe = make_probe(kind).fit(Xtr_fp, tr.labels)
                base_rep = classification_report(te.labels, probe.predict_proba(Xte_fp),
                                                 cfg.threshold)
                lo, hi = bootstrap_ci(te.labels, probe.predict_proba(Xte_fp),
                                      n=cfg.bootstrap, seed=cfg.seed)
                p_fp_te = probe.predict_proba(Xte_fp)

                for spec in specs:
                    Xtr_q = acts[spec.name][layer][:n_tr]
                    Xte_q = acts[spec.name][layer][n_tr:]
                    p_q = probe.predict_proba(Xte_q)
                    rep = classification_report(te.labels, p_q, cfg.threshold)
                    row = {
                        "model": backend.name,
                        "task": task, "layer": int(layer), "probe": kind,
                        "spec": spec.name, "spec_tags": list(spec.tags),
                        "baseline_auroc": base_rep["auroc"],
                        "baseline_auroc_lo": lo, "baseline_auroc_hi": hi,
                        "auroc": rep["auroc"], "auprc": rep["auprc"], "acc": rep["acc"],
                        "ece": rep["ece"], "brier": rep["brier"],
                        "delta_auroc": rep["auroc"] - base_rep["auroc"],
                        "n_test": rep["n"], "n_train": n_tr,
                    }
                    d_lo, d_hi = ((0.0, 0.0) if spec.is_baseline else
                                  paired_delta_ci(te.labels, p_fp_te, p_q,
                                                  n=cfg.bootstrap, seed=cfg.seed))
                    row["delta_lo"], row["delta_hi"] = d_lo, d_hi
                    # Significant only if the paired CI on the delta excludes 0.
                    row["delta_sig"] = bool(d_hi < 0 or d_lo > 0)
                    row.update({f"act_{k}": v for k, v in
                                activation_drift(Xte_fp, Xte_q).items()})
                    row.update({f"score_{k}": v for k, v in
                                score_drift(p_fp_te, p_q, cfg.threshold).items()})

                    if spec.is_baseline:
                        row.update(auroc_match=rep["auroc"], auroc_affine=rep["auroc"],
                                   auroc_refit=rep["auroc"], ece_match=rep["ece"],
                                   ece_affine=rep["ece"], ece_refit=rep["ece"],
                                   flip_match=0.0)
                    else:
                        m = recalibrate_match(probe, Xtr_fp[cal_idx], Xtr_q[cal_idx])
                        a = recalibrate_affine(probe, Xtr_q[cal_idx], tr.labels[cal_idx])
                        r = refit_on_quantized(kind, Xtr_q, tr.labels)
                        pm, pa, pr = (m.predict_proba(Xte_q), a.predict_proba(Xte_q),
                                      r.predict_proba(Xte_q))
                        row.update(
                            auroc_match=auroc(te.labels, pm),
                            auroc_affine=auroc(te.labels, pa),
                            auroc_refit=auroc(te.labels, pr),
                            ece_match=classification_report(te.labels, pm)["ece"],
                            ece_affine=classification_report(te.labels, pa)["ece"],
                            ece_refit=classification_report(te.labels, pr)["ece"],
                            flip_match=score_drift(p_fp_te, pm, cfg.threshold)["flip_rate"],
                        )
                    rows.append(row)

    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (d / "config.json").write_text(json.dumps({
            **{k: v for k, v in cfg.__dict__.items()},
            "layers": layers, "backend_name": backend.name,
            "specs": [s.to_dict() for s in specs],
        }, indent=2, default=str))
    return rows


def load_results(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
