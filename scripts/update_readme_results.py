#!/usr/bin/env python3
"""Inject the headline numbers + main table from a sweep into README.md.

Keeps the README's claims mechanically tied to a committed results.jsonl
instead of to whatever the numbers were when someone last wrote prose.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lens.experiment import load_results
from lens.report import headline, main_table

MARK = "<!-- RESULTS -->"
END = "<!-- /RESULTS -->"


def main(results="results/synthetic/results.jsonl", readme="README.md") -> int:
    rows = load_results(results)
    block = "\n".join([MARK, "", headline(rows), "",
                       "Logistic probe, averaged over 3 tasks x 4 layers "
                       "(`*` = paired bootstrap CI on the delta excludes zero):", "",
                       main_table(rows, "logistic"), "", END])
    text = Path(readme).read_text()
    start = text.index(MARK)
    stop = text.index(END) + len(END) if END in text else start + len(MARK)
    Path(readme).write_text(text[:start] + block + text[stop:])
    print(f"injected {len(rows)} rows into {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
