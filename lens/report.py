"""Turn sweep rows into markdown tables. No plotting dependency on purpose:
the results are small, and a table you can diff in a PR beats a PNG."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:.{nd}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def aggregate(rows: Iterable[dict], keys=("spec", "probe")) -> list[dict]:
    """Mean over everything not in `keys` (tasks and layers, normally)."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    num = ("baseline_auroc", "auroc", "delta_auroc", "auroc_match", "auroc_affine",
           "auroc_refit", "ece", "ece_match", "ece_refit", "score_flip_rate",
           "flip_match", "score_pearson_r", "act_cos_mean", "act_rel_l2",
           "delta_lo", "delta_hi", "delta_sig")
    out = []
    for k, g in groups.items():
        rec = dict(zip(keys, k))
        rec["n_rows"] = len(g)
        for c in num:
            vals = [r[c] for r in g if c in r and r[c] == r[c]]
            rec[c] = mean(vals) if vals else float("nan")
        out.append(rec)
    return sorted(out, key=lambda r: (r.get("probe", ""), -r["auroc"]))


def main_table(rows: list[dict], probe: str = "logistic") -> str:
    agg = [r for r in aggregate(rows) if r["probe"] == probe]
    agg.sort(key=lambda r: -r["auroc"])
    body = [[r["spec"], _fmt(r["baseline_auroc"]), _fmt(r["auroc"]),
             _fmt(r["delta_auroc"]) + ("*" if r.get("delta_sig", 0) > 0.5 else ""),
             _fmt(r["act_rel_l2"]), _fmt(r["score_pearson_r"]),
             _fmt(r["score_flip_rate"]), _fmt(r["ece"]),
             _fmt(r["ece_match"]), _fmt(r["flip_match"]), _fmt(r["auroc_refit"])] for r in agg]
    # No "AUROC +match" column on purpose: affine recalibration is monotone in the
    # logit, so it cannot change a rank-based metric. Its payoff is calibration
    # (ECE) and decision stability (flip rate), which is what is shown instead.
    return _table(["serving config", "AUROC fp16", "AUROC", "ΔAUROC", "act rel-L2",
                   "score r", "flip rate", "ECE", "ECE +match", "flip +match",
                   "AUROC +refit"], body)


def per_task_table(rows: list[dict], probe: str = "logistic") -> str:
    agg = aggregate([r for r in rows if r["probe"] == probe], keys=("task", "spec"))
    tasks = sorted({r["task"] for r in agg})
    specs = sorted({r["spec"] for r in agg})
    idx = {(r["task"], r["spec"]): r for r in agg}
    body = []
    for s in specs:
        row = [s] + [_fmt(idx[(t, s)]["delta_auroc"]) if (t, s) in idx else "-" for t in tasks]
        body.append(row)
    return _table(["serving config"] + [f"Δ {t}" for t in tasks], body)


def layer_table(rows: list[dict], probe: str = "logistic") -> str:
    agg = aggregate([r for r in rows if r["probe"] == probe
                     and "baseline" not in r["spec_tags"]], keys=("layer",))
    body = [[str(r["layer"]), _fmt(r["baseline_auroc"]), _fmt(r["delta_auroc"]),
             _fmt(r["act_rel_l2"]), _fmt(r["score_flip_rate"])] for r in
            sorted(agg, key=lambda r: r["layer"])]
    return _table(["layer", "AUROC fp16", "mean ΔAUROC (quantized)", "act rel-L2",
                   "flip rate"], body)


def probe_compare_table(rows: list[dict]) -> str:
    agg = aggregate([r for r in rows if "baseline" not in r["spec_tags"]], keys=("probe",))
    body = [[r["probe"], _fmt(r["baseline_auroc"]), _fmt(r["auroc"]), _fmt(r["delta_auroc"]),
             _fmt(r["score_flip_rate"]), _fmt(r["flip_match"]), _fmt(r["auroc_refit"])]
            for r in agg]
    return _table(["probe", "AUROC fp16", "AUROC quantized", "ΔAUROC", "flip rate",
                   "flip +match", "AUROC +refit"], body)


def headline(rows: list[dict]) -> str:
    """The three sentences a reader actually needs, computed from the rows."""
    q = [r for r in rows if "baseline" not in r["spec_tags"]]
    if not q:
        return "_no quantized configs in this sweep_"
    worst = min(q, key=lambda r: r["delta_auroc"])
    agg = aggregate(q, keys=("spec",))
    w4 = [r for r in agg if r["spec"].startswith("w4")]
    kv = [r for r in agg if r["spec"].startswith("kv")]
    lines = [
        f"- Mean ΔAUROC across all quantized configs: **{_fmt(mean([r['delta_auroc'] for r in q]))}** "
        f"(mean flip rate at the deployed threshold: **{_fmt(mean([r['score_flip_rate'] for r in q]))}**).",
        f"- Worst single cell: `{worst['spec']}` on `{worst['task']}` layer {worst['layer']} "
        f"({worst['probe']}), ΔAUROC **{_fmt(worst['delta_auroc'])}**.",
    ]
    if w4:
        lines.append(f"- Weight-only INT4 configs: mean ΔAUROC {_fmt(mean([r['delta_auroc'] for r in w4]))}, "
                     f"mean flip rate {_fmt(mean([r['score_flip_rate'] for r in w4]))}.")
    if kv:
        lines.append(f"- KV-cache-only configs: mean ΔAUROC {_fmt(mean([r['delta_auroc'] for r in kv]))}, "
                     f"mean flip rate {_fmt(mean([r['score_flip_rate'] for r in kv]))}.")
    sig = [r for r in q if r.get("delta_sig")]
    lines.append(f"- {len(sig)}/{len(q)} quantized cells show a ΔAUROC whose paired "
                 f"95% bootstrap CI excludes zero; the rest are inside the noise floor.")
    lines.append(
        f"- Repairs: label-free affine matching cuts mean flip rate "
        f"{_fmt(mean([r['score_flip_rate'] for r in q]))} -> {_fmt(mean([r['flip_match'] for r in q]))}; "
        f"retraining on quantized activations moves mean AUROC "
        f"{_fmt(mean([r['auroc'] for r in q]))} -> {_fmt(mean([r['auroc_refit'] for r in q]))}.")
    return "\n".join(lines)


def build_report(rows: list[dict], title: str = "LENS sweep") -> str:
    probes = sorted({r["probe"] for r in rows})
    parts = [f"# {title}", "",
             f"{len(rows)} rows | tasks: {', '.join(sorted({r['task'] for r in rows}))} "
             f"| layers: {sorted({r['layer'] for r in rows})} | probes: {', '.join(probes)}",
             "", "## Probe families under quantization", "", probe_compare_table(rows), ""]
    for p in probes:
        parts += [f"## {p} probe: degradation by serving config", "",
                  main_table(rows, p), "",
                  f"### {p}: ΔAUROC per task", "", per_task_table(rows, p), ""]
    parts += ["## Layer sensitivity (mean over quantized configs)", "",
              layer_table(rows, probes[0]), "", "## Headline numbers", "",
              headline(rows), "",
              "Columns: ΔAUROC is quantized minus FP16 for the *unchanged* FP16-trained "
              "probe, marked `*` when its paired bootstrap CI excludes zero. "
              "`flip rate` is the fraction of test decisions that change at the "
              "deployed threshold -- the operational cost AUROC cannot see. `+match` is "
              "label-free affine recalibration on paired activations; `+refit` retrains "
              "on quantized activations -- the strongest linear repair available, though "
              "not guaranteed to help: a direction fit on degraded activations can be "
              "worse than one transferred from clean ones."]
    return "\n".join(parts)
