#!/usr/bin/env bash
# Phase 1 on Colab/Kaggle: write the split files, then train and score the classical baseline.
#   bash scripts/run_phase1.sh /path/to/RSCD               # tune on validation only
#   bash scripts/run_phase1.sh /path/to/RSCD --eval-test   # final run, scores the test set
# Colab and Kaggle already ship numpy, pillow, pyyaml, scikit-learn and scikit-image.
set -euo pipefail
cd "$(dirname "$0")/.."

root="${1:?usage: bash scripts/run_phase1.sh /path/to/RSCD [--eval-test]}"
shift

if [[ -f data/splits/rscd_train_subset.csv ]]; then
  echo "Split files exist in data/splits; skipping prepare (delete them to rebuild)."
else
  python scripts/prepare_rscd.py --root "$root"
fi
python scripts/train_baseline.py --root "$root" "$@"
