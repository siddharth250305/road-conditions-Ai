# Road Condition AI

Road surface (friction) classification on RSCD and a cross-country damage detection study on RDD2022.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,classical]"   # add ,notebook / ,train / ,detect / ,export / ,serve as needed
pytest
```

Datasets go in `data/raw/` and are not committed. Saved split file lists go in `data/splits/` and are committed.

## Phase 1: RSCD data and classical baseline (C1, C2, C5)

| Step | Command | Output |
|---|---|---|
| Split files | `python scripts/prepare_rscd.py --root <RSCD>` | `data/splits/rscd_{train_subset,val,test}.csv`, `rscd_leak_report.json` |
| Tune baseline | `python scripts/train_baseline.py --root <RSCD>` | validation table, `artifacts/results/baseline.json` |
| Final run | `python scripts/train_baseline.py --root <RSCD> --eval-test` | test table (run once per final model) |
| EDA | `notebooks/01_rscd_eda.ipynb` with `RSCD_ROOT=<RSCD>` | class counts, sample grid, image stats, split checks |

- The train subset is 40,000 images, balanced over the 6 friction classes and spread over the
  material/unevenness sub-labels, drawn once with seed 0. Official val and test are listed, not sampled.
- A train image within 4 bits (dHash) of any val/test image is dropped and replaced from the
  same class; the leak report lists counts and example pairs to inspect in the notebook.
- The split folder names (`train`, `vali_20k`, `test_50k`) and the label parsing are assumptions
  until checked against the real download; change `rscd.split_dirs` / `rscd.label_source` in
  `configs/default.yaml` if they differ. Unknown labels stop the run with an error.

**Local dry run on mock data** (no download needed):

```bash
python -m roadcond.data.mock --out data/raw/rscd_mock
python scripts/prepare_rscd.py --root data/raw/rscd_mock --splits-dir data/processed/mock_splits --train-size 120
python scripts/train_baseline.py --root data/raw/rscd_mock --splits-dir data/processed/mock_splits --eval-test
```

**Google Colab** (numpy, pillow, pyyaml, scikit-learn and scikit-image are preinstalled):

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone https://github.com/siddharth250305/road-conditions-Ai.git
%cd road-conditions-Ai
# Reading ~110k small files from Drive is slow: unzip to local disk first.
!unzip -q /content/drive/MyDrive/RSCD.zip -d /content/rscd
!bash scripts/run_phase1.sh /content/rscd               # tune on validation
!bash scripts/run_phase1.sh /content/rscd --eval-test   # final test run
```

**Kaggle**: add the dataset as notebook input, then run the same `git clone` and
`!bash scripts/run_phase1.sh /kaggle/input/<dataset-folder>`. Inputs are read-only; all
outputs land in the clone under `/kaggle/working`.

Afterwards, commit `data/splits/` back to the repo so every later run uses the same files.
Feature caches (`data/processed/`) and results (`artifacts/`) stay out of git.
