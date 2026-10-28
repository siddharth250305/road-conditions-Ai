#!/usr/bin/env bash
# Phase 2 on Colab/Kaggle: train and score the three classifiers on the saved RSCD split files.
#   bash scripts/run_phase2.sh /path/to/RSCD                       # tune on validation only
#   bash scripts/run_phase2.sh /path/to/RSCD --eval-test --gradcam 4
#   bash scripts/run_phase2.sh /path/to/RSCD --resume              # continue after a timeout
# Extra arguments go to scripts/train_classifier.py; do not pass --model (all three are run).
# Needs a GPU runtime and data/splits/ from scripts/run_phase1.sh (or committed to the repo).
set -euo pipefail
cd "$(dirname "$0")/.."

root="${1:?usage: bash scripts/run_phase2.sh /path/to/RSCD [train_classifier.py args]}"
shift

if [[ ! -f data/splits/rscd_train_subset.csv ]]; then
  echo "data/splits/ is missing: run scripts/run_phase1.sh first, or commit the split files." >&2
  exit 1
fi
# Colab already ships torch, torchvision, numpy, pillow and matplotlib.
pip install -q timm grad-cam pyyaml

for model in small_cnn resnet50 efficientnet; do
  echo "=== $model ==="
  python scripts/train_classifier.py --model "$model" --root "$root" "$@"
done
