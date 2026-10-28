#!/usr/bin/env bash
# Phase 3 on Colab/Kaggle: the whole cross-country study, one step after another.
#   bash scripts/run_phase3.sh /path/to/RDD2022
#   bash scripts/run_phase3.sh /path/to/RDD2022 --resume      # continue after a timeout
#   bash scripts/run_phase3.sh /path/to/RDD2022 --epochs 2 --imgsz 320    # quick dry run
# Extra arguments go to every step (do not pass --augment). Finished work is skipped, so
# rerun the same command after a disconnect. Needs a GPU runtime for real settings.
set -eo pipefail
cd "$(dirname "$0")/.."

root="${1:?usage: bash scripts/run_phase3.sh /path/to/RDD2022 [extra args for every step]}"
shift

# Colab ships torch; Ultralytics (AGPL-3.0) and Albumentations are installed on demand.
pip install -q ultralytics albumentations pyyaml

run() { echo "=== $*"; python scripts/train_detection.py "$@"; }

if [[ -f data/splits/rdd_japan_pool.txt && -d data/processed/rdd_yolo/Japan/labels ]]; then
  echo "Split files and work tree exist; skipping prepare (delete data/processed/rdd_yolo to rebuild)."
else
  run prepare --root "$root" "$@"
fi
run source "$@"
run source --augment "$@"
run fewshot "$@"
run pseudo "$@"
run report "$@"
