#!/usr/bin/env python3
"""Inject the headline numbers + main table from a sweep into README.md.

Keeps the README's claims mechanically tied to a committed results.jsonl
instead of to whatever the numbers were when someone last wrote prose.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lens.experiment import load_results
from lens.report import headline, main_table, task_summary_table

BLOCKS = {
    "RESULTS": ("results/synthetic/results.jsonl",
                "Logistic probe, averaged over 3 tasks x 4 layers "
                "(`*` = paired bootstrap CI on the delta excludes zero):"),
    "REAL-RESULTS": ("results/qwen2.5-0.5b/results.jsonl",
                     "Logistic probe, averaged over 3 tasks x 4 layers "
                     "(`*` = paired bootstrap CI on the delta excludes zero):"),
}


def inject(text: str, name: str, block: str) -> str:
    mark, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    start = text.index(mark)
    stop = text.index(end) + len(end) if end in text else start + len(mark)
    return text[:start] + "\n".join([mark, "", block, "", end]) + text[stop:]


def main(readme="README.md") -> int:
    text = Path(readme).read_text()
    for name, (path, caption) in BLOCKS.items():
        if not Path(path).exists():
            print(f"skip {name}: {path} missing")
            continue
        rows = load_results(path)
        parts = [headline(rows), "", caption, "", main_table(rows, "logistic")]
        if name == "REAL-RESULTS":
            parts = [headline(rows), "",
                     "Per task, and whether there was any headroom to lose:", "",
                     task_summary_table(rows), "", caption, "",
                     main_table(rows, "logistic")]
        text = inject(text, name, "\n".join(parts))
        print(f"injected {len(rows)} rows for {name}")
    Path(readme).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
