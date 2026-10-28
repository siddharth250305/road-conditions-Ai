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

## Phase 2: deep classifiers (C3, C4, C8, C9)

Small CNN (from scratch), fine-tuned ResNet-50 and EfficientNet-B0 (timm), trained on the saved
split files from Phase 1. Settings live in the `classifier:` section of `configs/default.yaml`.

| Model | Epochs / patience | Notes |
|---|---|---|
| `small_cnn` | 30 / 6 | 4 x (Conv, BatchNorm, ReLU, MaxPool, Dropout), global average pooling |
| `resnet50`, `efficientnet` | 12 / 4 | head-only for 1 epoch, then the whole network at a 10x smaller backbone learning rate |

AdamW, cosine annealing, early stopping on validation macro-F1, mixed precision on GPU. A checkpoint
is written every epoch, so after a Colab timeout rerun the same command with `--resume`.
After training, a temperature is fitted on **validation** logits; ECE before and after scaling is
reported for validation and (with `--eval-test`) for test. Judge H4 on the test numbers.

```bash
python scripts/train_classifier.py --model resnet50 --root <RSCD>                     # tune on val
python scripts/train_classifier.py --model resnet50 --root <RSCD> --eval-test --gradcam 4
```

Outputs in `artifacts/` (git-ignored): `checkpoints/<model>_seed<n>/{best,last}.pt`, `history.json`,
`loss_curves.png`, `reliability_<split>.png`, `gradcam_<split>.png`, and `results/<model>.json`
(same shape as `baseline.json`, so the H1 table can compare all models).

**Google Colab** (use a GPU runtime; torch and torchvision are preinstalled):

```python
!git clone https://github.com/siddharth250305/road-conditions-Ai.git
%cd road-conditions-Ai
!unzip -q /content/drive/MyDrive/RSCD.zip -d /content/rscd   # local disk, not Drive
# data/splits/ must exist: run scripts/run_phase1.sh once, or commit the split files.
!bash scripts/run_phase2.sh /content/rscd                        # all three models, val only
!bash scripts/run_phase2.sh /content/rscd --eval-test --gradcam 4   # final run, once
!bash scripts/run_phase2.sh /content/rscd --resume               # continue after a timeout
```

Copy `artifacts/` to Drive before the session ends. To run one model, call
`scripts/train_classifier.py --model <name>` directly. Use `--limit 500 --epochs 1` for a quick
dry run before the full job.

Local check without any download: `pytest tests/test_dl.py` (needs `pip install -e ".[dev,train]"`).

## Phase 3: cross-country damage detection (D1-D5)

A YOLOv8 detector trained on one country (Japan) and scored on others (Norway, Czech Republic,
USA), then three ways to recover the lost accuracy. Install with `pip install -e ".[dev,detect]"`.
**Ultralytics is AGPL-3.0**: it stays in the optional `detect` extra; decide the repository
licence before making it public.

| Step | Command | PRD |
|---|---|---|
| Split countries, write lists and data configs | `prepare --root <RDD2022>` | data rules |
| Train on Japan, score every country | `source` | D1 |
| Same with weather augmentation, no target images | `source --augment` | D2 |
| Fine-tune on 10 / 50 / 200 target labels, 3 seeds | `fewshot` | D3, D5 |
| Pseudo-label unlabelled target images, retrain | `pseudo` | D4 |
| Tables, learning curve, `detection_report.md` | `report` | D1-D5 |

Run each as `python scripts/train_detection.py <step> [options]`, or all of them with
`bash scripts/run_phase3.sh <RDD2022>`. Finished work is skipped, so after a Colab timeout rerun
the same command (add `--resume` to continue a half-trained source detector).

- **Splits.** RDD2022 test labels are not public, so each country gets its own seeded
  70 / 10 / 20 % train-pool / val / test split from its labelled images; pool images that nearly
  duplicate a val/test image are dropped. The exact lists are `data/splits/rdd_<country>_*.txt`
  (commit them). The generated `data/splits/rdd_*.yaml` hold absolute paths and are git-ignored.
- **Recipe.** Source: YOLOv8n, 640 px, 30 epochs, at most 5,000 Japanese images (a cap for free
  Colab; `--max-train-images` changes it). Few-shot: fixed recipe, backbone frozen, final weights
  scored, never tuned on target data; the 10 images are inside the 50, inside the 200, per seed.
- **Metrics.** mAP@0.5 and mAP@0.5:0.95 from Ultralytics; F1 at a fixed confidence of 0.25 from
  our own matcher. Relative gap = (source - target) / source; "gap recovered" is `n/a` when the
  target is not worse than the source.
- **Test split** of each country is scored once per final model. Pseudo-label thresholds
  {0.3, 0.5, 0.7} are chosen on target *validation* images, which is the one place target labels
  influence a label-free method; the results file says so.

**Google Colab** (GPU runtime; torch is preinstalled):

```python
!git clone https://github.com/siddharth250305/road-conditions-Ai.git
%cd road-conditions-Ai
# Put the RDD2022 download on local disk first, e.g. unzip it to /content/rdd2022
!bash scripts/run_phase3.sh /content/rdd2022
!bash scripts/run_phase3.sh /content/rdd2022 --resume      # after a disconnect
```

Copy `artifacts/` and `data/splits/rdd_*.txt` to Drive before the session ends. Dry run first:
`bash scripts/run_phase3.sh /content/rdd2022 --epochs 1 --imgsz 320`.

Local check without any download: `pytest tests/test_detection.py` (39 tests, about 10 seconds,
including a real Ultralytics run on a synthetic mock dataset).

## Phase 4: ONNX export and demo API (W1, W2, W3)

A trained classifier becomes a small **bundle** (`model.onnx`, `meta.json`, optionally
`study.json`), and a FastAPI service answers from that folder alone. Install with
`pip install -e ".[dev,export,serve]"`; the server itself needs only `.[serve]` (no torch).

```bash
# 1. Export: PyTorch checkpoint -> one ONNX file, checked against PyTorch (tolerance 1e-4)
python scripts/export_model.py bundle --checkpoint <best.pt> --results <results/model.json> --out serving
# 2. The cross-country benchmark for the demo's study view (from the Phase 3 result files)
python scripts/export_model.py study --results-dir artifacts/results --out serving/study.json
# 3. Serve
ROADCOND_BUNDLE_DIR=serving uvicorn api.main:app --port 8000      # docs at /docs
curl -F file=@road.jpg localhost:8000/predict
```

| Endpoint | Returns |
|---|---|
| `POST /predict` (multipart `file`, optional `?heatmap=false`) | `prediction`, calibrated `confidence`, `is_uncertain` + `display_label` ("Not sure"), `top2`, all `probabilities`, `heatmap_base64` (PNG overlay), `limits_note` |
| `GET /study` | slope-chart points (source vs target mAP) and the few-shot learning curve, from your measured results only; 404 if absent |
| `GET /health` | status, model, classes; 503 if no model is loaded |

- **The ONNX graph contains the normalisation**, so the server only resizes the picture (Pillow,
  no torch). Its second output is a per-class activation map: for SmallCNN, ResNet-50 and
  EfficientNet-B0 this equals Grad-CAM at the last feature map (a test checks it), so heatmaps
  need no gradients at serving time.
- **Confidence is calibrated:** softmax of logits divided by the temperature stored in `meta.json`
  (fitted in Phase 2 on validation data). "Not sure" appears when it is below 0.5
  (`ROADCOND_UNCERTAINTY_THRESHOLD`; the PRD gives no value).
- **Errors:** 400 empty or corrupt files, 413 too many bytes (10 MB) or pixels (50 million),
  422 missing `file`, 503 no model. Settings: `ROADCOND_BUNDLE_DIR`, `ROADCOND_MAX_UPLOAD_MB`,
  `ROADCOND_MAX_PIXELS`, `ROADCOND_HEATMAP_ALPHA`, `ROADCOND_CORS_ORIGINS` (only for a separate
  dev frontend).
- **A bundle built from `--mock` data says so** in `/health` and in the limits note.
- `.onnx` files are git-ignored. ResNet-50's is about 94 MB: ship it through Git LFS or the
  Hugging Face Hub, not a plain commit.

Local check without any download or training: `pytest tests/test_export.py tests/test_api.py`
(48 tests, about 20 seconds).
