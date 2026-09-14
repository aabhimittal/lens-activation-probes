#!/usr/bin/env bash
# End-to-end demo: no GPU, no downloads, ~2 minutes on 4 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m lens.cli sweep --config experiments/configs/synthetic.yaml --out results/synthetic
python -m lens.cli export --config experiments/configs/synthetic.yaml \
    --tasks injection --layers 5 --probes logistic --bundle artifacts/injection_l5
echo
echo "report:  results/synthetic/report.md"
echo "bundle:  artifacts/injection_l5.{npz,json}"
