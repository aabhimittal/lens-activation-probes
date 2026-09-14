#!/usr/bin/env python3
"""Does quantization change *which* questions the model gets wrong?

The sweep freezes labels at FP16 on purpose, so that a change in AUROC means
probe degradation and nothing else. That isolation is correct for measuring the
probe, and it quietly assumes away the other half of the deployment question:
in production the quantized model is the one generating, and if it fails on a
different set of inputs, a perfectly robust probe is still describing a system
that no longer exists.

This script measures that second effect directly: same questions, answers
regenerated under each serving config, graded the same way.

    python scripts/measure_label_drift.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --n 400 --specs w4-g128 kv3 w4-g128+kv4 --out results/label_drift.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lens.backends import get_backend
from lens.generate import format_prompt, label_drift, load_triviaqa
from lens.quant import get_specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--specs", nargs="+", default=["w4-g128", "kv3", "w4-g128+kv4"])
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=16)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=24)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", default="results/label_drift.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    be = get_backend("hf", model_id=a.model, device=a.device, dtype=a.dtype,
                     batch_size=a.batch_size, max_len=256)
    questions, aliases = load_triviaqa(a.n, a.seed)
    prompts = [format_prompt(be.tok, q) for q in questions]
    rows = label_drift(be, prompts, aliases, get_specs(a.specs), a.max_new_tokens)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "n": len(prompts), "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
