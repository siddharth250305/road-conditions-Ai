# Road Condition AI: System Guidance & Verification Manual

This document explains what has been built in the **Road Condition AI** repository, how each component works, and how you can check every requirement yourself against the [PRD](Road%20Condition%20AI%20PRD.pdf). Every command in section 4 was run on the mock dataset, and the outputs shown are the real ones.

---

## Table of Contents
1. [Executive Summary & Current Status](#1-executive-summary--current-status)
2. [PRD Requirements Coverage Matrix](#2-prd-requirements-coverage-matrix)
3. [What Was Built (Component Deep Dive)](#3-what-was-built-component-deep-dive)
   - [3.1 Configuration & Package Setup](#31-configuration--package-setup)
   - [3.2 The Anti-Leak RSCD Data Pipeline](#32-the-anti-leak-rscd-data-pipeline)
   - [3.3 The Synthetic Mock Data Engine](#33-the-synthetic-mock-data-engine)
   - [3.4 The Pure-NumPy Metric Engine](#34-the-pure-numpy-metric-engine)
   - [3.5 The Classical Machine Learning Baseline](#35-the-classical-machine-learning-baseline)
   - [3.6 PyTorch Dataset & Transforms](#36-pytorch-dataset--transforms)
   - [3.7 SmallCNN (Req C3)](#37-smallcnn-req-c3)
   - [3.8 Pretrained ResNet-50 & EfficientNet-B0 (Req C4)](#38-pretrained-resnet-50--efficientnet-b0-req-c4)
   - [3.9 Training Engine & Early Stopping](#39-training-engine--early-stopping)
   - [3.10 Temperature Scaling & Calibration (Req C9)](#310-temperature-scaling--calibration-req-c9)
   - [3.11 Grad-CAM Explainability (Req C8)](#311-grad-cam-explainability-req-c8)
   - [3.12 Standalone Cloud Scripts](#312-standalone-cloud-scripts)
   - [3.13 Exploratory Data Analysis (EDA) Notebook](#313-exploratory-data-analysis-eda-notebook)
4. [Step-by-Step Manual Verification Playbook](#4-step-by-step-manual-verification-playbook)
   - Phase 1: [Tests 1-6](#phase-1-data-pipeline-metrics-and-classical-baseline)
   - Phase 2: [Tests 7-13](#phase-2-deep-learning-classifier)
5. [Reading the Results Against the PRD Hypotheses](#5-reading-the-results-against-the-prd-hypotheses)
6. [Running the Real Experiments on Google Colab](#6-running-the-real-experiments-on-google-colab)
7. [Known Limitations & Assumptions](#7-known-limitations--assumptions)
8. [What Remains to Be Built (Next Phases)](#8-what-remains-to-be-built-next-phases)

---

## 1. Executive Summary & Current Status

The repository contains the completed code for **Phase 0 (scaffolding)**, **Phase 1 (RSCD data pipeline and classical baseline)** and **Phase 2 (deep-learning surface classifier)**.

**What is verified:** all 67 automated tests pass on the mock dataset (CPU only), `ruff` is clean on `src`, `tests` and `scripts`, and every command in section 4 runs. A real pretrained EfficientNet-B0 (weights downloaded from Hugging Face) has been run end to end through the command-line script on mock data.

**What is not verified yet:** no model has been trained on the real RSCD data, so there are **no real accuracy numbers**. The GPU and mixed-precision (AMP) code path has not been run, because it needs a CUDA machine. The assumed RSCD folder layout (section 7) still has to be checked on the real download. Mock scores mean nothing: the mock classes differ only by colour.

### Why this foundation matters
1. **A leakage-free, balanced data subsetter** stops near-identical consecutive frames from inflating scores.
2. **A pure-NumPy evaluation system** reports macro-F1 instead of misleading raw accuracy.
3. **A classical baseline** (colour histograms + HOG into logistic regression and random forest) sets the bar the deep models must beat (hypothesis H1).
4. **Three deep classifiers with the same evaluation**, plus calibration and Grad-CAM, so the models can be compared fairly and their confidence and attention can be inspected.

---

## 2. PRD Requirements Coverage Matrix

| PRD Req ID | Description | Status | Files | How to Verify |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | EDA notebook: class counts, sample grid, image stats, split sanity checks | Complete | [`notebooks/01_rscd_eda.ipynb`](notebooks/01_rscd_eda.ipynb) | Open in Jupyter or Colab and run all cells |
| **C2** | Classical baseline: colour histogram + HOG into logistic regression & random forest | Complete | [`models/baseline.py`](src/roadcond/models/baseline.py) | Test 5 |
| **C3** | Small CNN written from scratch | Code complete, mock-verified; real run pending | [`models/small_cnn.py`](src/roadcond/models/small_cnn.py), [`train.py`](src/roadcond/train.py) | Tests 7, 9 |
| **C4** | Fine-tuned pretrained ResNet-50 / EfficientNet | Code complete, mock-verified; real run pending | [`models/pretrained.py`](src/roadcond/models/pretrained.py) | Tests 8, 9 |
| **C5** | Evaluation report: macro-F1, per-class precision/recall, confusion matrix | Complete | [`metrics.py`](src/roadcond/metrics.py) | Tests 1, 4 |
| **C6** | Imbalance handling compared (none / class-weighted / oversampling) | Not built | | |
| **C7** | Hyperparameter sweep with an experiment tracker | Not built (learning rate, freeze epochs and augmentation strength are in the config, ready for it) | | |
| **C8** | Grad-CAM heatmaps for correct and wrong predictions | Code complete, mock-verified; real run pending | [`explain.py`](src/roadcond/explain.py) | Test 10 |
| **C9** | Calibration: reliability diagram, ECE before and after temperature scaling | Code complete, mock-verified; real run pending | [`calibration.py`](src/roadcond/calibration.py) | Test 11 |
| **C10** | Optional ViT / DINOv2 comparison | Not built (P2) | | |
| **Data Rules** | Balanced 40k subset, official val/test untouched, near-duplicate rejection | Complete | [`data/rscd.py`](src/roadcond/data/rscd.py) | Test 3 |
| **D1-D5** | RDD2022 YOLO cross-country study | Next phase | | |
| **W1-W3** | Demo (FastAPI + React/D3) | Later phase | | |

---

## 3. What Was Built (Component Deep Dive)

### 3.1 Configuration & Package Setup
- **[`pyproject.toml`](pyproject.toml)**: `hatchling` build backend, Python `>=3.11`, `src/` layout. The core install is light (`numpy`, `pillow`, `pyyaml`); heavy libraries are optional extras:
  - `classical`: `scikit-learn`, `scikit-image`, `joblib`
  - `train`: `roadcond[classical]`, `torch`, `torchvision`, `timm`, `grad-cam`, `pandas`, `matplotlib`, `tqdm`, `tensorboard`
  - `detect`: `ultralytics`, `albumentations` (kept separate: Ultralytics is AGPL-3.0)
  - `export`: `onnx`, `onnxruntime`; `serve`: `fastapi`, `uvicorn`, `python-multipart`, `pillow`, `onnxruntime`
  - `notebook`: `jupyterlab`, `matplotlib`; `dev`: `pytest`, `ruff`, `nbstripout`
- **[`configs/default.yaml`](configs/default.yaml)**: the single source of truth for settings:
  - Friction classes, ordered from dry to ice: `[dry, wet, water, fresh_snow, melted_snow, ice]`; image size `[240, 360]` (height, width).
  - Training subset: 40,000 images, fixed `seed: 0`. Dedup: 64-bit dHash, max Hamming distance 4.
  - `baseline:` feature sizes and the logistic regression / random forest grids.
  - `classifier:` batch size 64, workers 2, mixed precision on, weight decay 0.01, `augment_strength` 1.0, `freeze_epochs` 1, 15 calibration bins, and per-model settings:

    | Model | Epochs | Patience | Learning rate | Backbone LR |
    | :--- | :--- | :--- | :--- | :--- |
    | `small_cnn` | 30 | 6 | 0.001 | n/a (dropout 0.1) |
    | `resnet50` | 12 | 4 | 0.001 (head) | 0.0001 |
    | `efficientnet` (`efficientnet_b0`) | 12 | 4 | 0.001 (head) | 0.0001 |
- **[`config.py`](src/roadcond/config.py)**: `load_config()` and `resolve_path()`, so relative paths work from the repo root or from inside scripts.

### 3.2 The Anti-Leak RSCD Data Pipeline
- **File**: [`src/roadcond/data/rscd.py`](src/roadcond/data/rscd.py)
- **Problem it solves**: consecutive frames from one drive look almost identical. If one lands in train and the next in test, the model scores well without learning friction.
- **How it works**:
  1. **Label mapping (`to_friction`)**: turns a full RSCD label such as `wet_concrete_slight` into its friction class (`wet`). An unknown label stops the run with an error instead of being guessed.
  2. **Balanced subset (`balanced_subset`)**: 40,000 images split over 6 classes gives quotas of 6,667 (dry, wet, water, fresh snow) and 6,666 (melted snow, ice). Inside a class, `interleave_by_label` round-robins over the material and roughness sub-labels, so the subset is not dominated by smooth asphalt.
  3. **Perceptual hash (`dhash`)**: a 64-bit difference hash computed with PIL and NumPy. Unlike MD5/SHA it stays similar when brightness or JPEG compression changes slightly.
  4. **Near-duplicate index (`NearDuplicateIndex`)**: avoids comparing every train image with every val/test image. The 64 bits are split into `max_hamming + 1` = **five bands**. Two hashes within 4 bits must agree exactly on at least one band (pigeonhole principle), so only images sharing a band value are compared.
  5. **Leak check (`LeakChecker`)**: rejects a train image within `max_hamming` bits of any val/test image. If `rscd.dedup.drive_id_regex` is set, it also rejects train images that share a drive id with a val/test image.
  6. **Replacement**: a rejected image is replaced by another from the same class, so the subset stays balanced. If a class runs out, the shortfall is logged and written to the report.
  7. **Outputs**: `data/splits/rscd_train_subset.csv`, `rscd_val.csv`, `rscd_test.csv` (columns `path, friction, label`) and the audit log `rscd_leak_report.json`.
- **Untouched splits**: val and test are only listed, never sampled. The report also counts test images that nearly duplicate val images, for information only.

### 3.3 The Synthetic Mock Data Engine
- **File**: [`src/roadcond/data/mock.py`](src/roadcond/data/mock.py)
- **Problem it solves**: the full RSCD set is about a million images, and tests should not need a download.
- **How it works**: writes small synthetic images (coloured noise, *not* realistic road photos) in the layout the config assumes: `train/<label>/*.jpg` and flat, timestamped `vali_20k` / `test_50k` folders. It deliberately plants 2 near-duplicates of test images in the training folder (`leak_000.jpg`, `leak_001.jpg`) so the leak check can be proven to work.

### 3.4 The Pure-NumPy Metric Engine
- **File**: [`src/roadcond/metrics.py`](src/roadcond/metrics.py)
- **Functions**:
  - `confusion_matrix(y_true, y_pred, n_classes)`: rows are true classes, columns are predictions. Classes that never occur keep a row of zeros. Labels outside `0..n_classes-1` raise `ValueError`.
  - `precision_recall_f1(cm)`: per-class precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). A zero denominator gives 0.0, matching scikit-learn's `zero_division=0`.
  - `macro_f1(y_true, y_pred, n_classes)`: the unweighted mean of per-class F1, so a weak rare class (ice) cannot hide behind a strong common one.
  - `accuracy(y_true, y_pred)`, `classification_report(y_true, y_pred, class_names)` (a JSON-ready dict with accuracy, macro-F1, per-class metrics and the confusion matrix).
  - `mean_std(values)` formats results over seeds as `mean ± std`; `format_table(headers, rows)` prints a Markdown table.

### 3.5 The Classical Machine Learning Baseline
- **File**: [`src/roadcond/models/baseline.py`](src/roadcond/models/baseline.py)
- **Features (`extract_features`)**: images are downscaled to 120 x 180, then:
  - 16-bin normalised histograms for each of R, G, B, H, S, V = **96** colour dimensions;
  - HOG on the grey image (9 orientations, 16 x 16 pixel cells, 2 x 2 cells per block) = **2,160** texture dimensions;
  - **2,256** dimensions in total.
- **Feature cache**: extraction runs in parallel (`ProcessPoolExecutor`) and is cached to `data/processed/rscd_features_{split}.npz`. The cache key covers the file list and settings, so a changed config recomputes it.
- **Models**: `StandardScaler` + logistic regression (`C` in {0.1, 1.0}) and a 300-tree random forest (`max_depth` in {None, 30}).
- **Integrity**: the grid is chosen on validation macro-F1 only. The test split is scored only with `--eval-test`.
- **Output**: `artifacts/results/baseline.json` (3 seeds: 0, 1, 2).

### 3.6 PyTorch Dataset & Transforms
- **File**: [`src/roadcond/data/dataset.py`](src/roadcond/data/dataset.py)
- `RoadDataset(records, root, classes, transform)` reads the saved split CSVs (via `load_splits`) and returns `(image tensor, class index)`. `load_raw(i, size)` returns the plain resized image for Grad-CAM overlays.
- `build_transforms(train, size, strength)` uses **torchvision v2**:
  - Training: resize to 240 x 360, random horizontal flip, mild colour jitter (brightness 0.2, contrast 0.2, saturation 0.1, scaled by `strength`), ImageNet normalisation. There is **no hue shift**: colour is a real cue for snow, ice and wet roads.
  - Validation / test: resize and normalise only (deterministic).
- `make_loader(...)` gives a seeded shuffle and drops the last partial batch in training (BatchNorm cannot train on a batch of one).

### 3.7 SmallCNN (Req C3)
- **File**: [`src/roadcond/models/small_cnn.py`](src/roadcond/models/small_cnn.py)
- Four stages of **Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout2d** with 32, 64, 128 and 256 channels, then global average pooling and one linear layer to 6 classes. **390,438 parameters.**
- ReLU is not in-place, because Grad-CAM hooks that layer. `model.target_layer` points at the last stage's ReLU, before the final pooling (the largest feature map).

### 3.8 Pretrained ResNet-50 & EfficientNet-B0 (Req C4)
- **File**: [`src/roadcond/models/pretrained.py`](src/roadcond/models/pretrained.py)
- `build_pretrained(timm_name, num_classes, pretrained, freeze_backbone)` creates the timm model; timm replaces its classifier with a new 6-class linear head. Pretrained weights download from the Hugging Face hub on first use.
- `set_backbone_trainable(model, bool)` freezes or unfreezes everything except the head. `freeze_batchnorm_stats(model)` keeps BatchNorm layers in eval mode while the backbone is frozen, so their running statistics are not disturbed. `split_parameters` and `count_parameters` support the optimiser groups and the checks below.
- **Fine-tuning schedule**: the first `freeze_epochs` (1) epoch trains only the head. After that the whole network trains, with the backbone at a 10x smaller learning rate than the head.

### 3.9 Training Engine & Early Stopping
- **File**: [`src/roadcond/train.py`](src/roadcond/train.py)
- AdamW, cosine annealing over the epoch budget, cross-entropy loss, mixed precision on CUDA only.
- **Early stopping** (`EarlyStopper`): stops when validation macro-F1 has not improved for `patience` epochs (a tie is not an improvement).
- **Checkpoints**: `best.pt` on every improvement, `last.pt` every epoch (written via a temporary file, so a Colab disconnect cannot corrupt it). `--resume` continues from `last.pt`.
- `run_seed` trains one model with one seed, reloads the best weights, fits the temperature, scores validation (and test with `--eval-test`), writes plots and the optional Grad-CAM gallery. `main` runs all seeds and prints the results table.
- **Command line**: `--model small_cnn|resnet50|efficientnet`, `--eval-test`, `--resume`, `--gradcam N`, `--seeds`, `--epochs`, `--batch-size`, `--workers`, `--freeze-epochs`, `--device`, `--no-pretrained` (random weights, no download), `--limit N` (debug: at most N images per split).

### 3.10 Temperature Scaling & Calibration (Req C9)
- **File**: [`src/roadcond/calibration.py`](src/roadcond/calibration.py)
- `TemperatureScaler` learns one number T > 0 (stored as log T, so it stays positive) and divides the logits by it. It is fitted with L-BFGS on the negative log-likelihood of **validation** logits. Scaling never changes which class is predicted, only the confidence.
- `expected_calibration_error` splits predictions into 15 equal-width confidence bins and averages |accuracy - confidence| weighted by bin size. `reliability_bins` returns the per-bin data, and `save_reliability_plot` draws the before and after diagrams.
- **How to read T**: T > 1 means the model was over-confident and gets softened; T < 1 means it was under-confident and gets sharpened.
- **Held-out caveat**: ECE after scaling *on the validation set* is optimistic, because T was fitted there. Judge hypothesis H4 on the **test** numbers.

### 3.11 Grad-CAM Explainability (Req C8)
- **File**: [`src/roadcond/explain.py`](src/roadcond/explain.py)
- `gradcam(model, inputs, target_layer, class_ids)` uses `pytorch-grad-cam` and returns heatmaps of shape `(batch, 240, 360)` with values in [0, 1]. `default_target_layer` picks the last convolutional stage (`SmallCNN.target_layer`, ResNet `layer4[-1]`, EfficientNet `blocks[-1]`). It runs in eval mode and works even when all parameters are frozen.
- `overlay(raw, cam)` blends a heatmap onto the image; `select_examples` picks the most confident **mistakes** and a seeded random sample of correct predictions; `save_gallery` draws original and overlay rows, titled `true -> predicted (confidence)`. The confidence shown is the calibrated one.

### 3.12 Standalone Cloud Scripts
- [`scripts/prepare_rscd.py`](scripts/prepare_rscd.py): builds the split CSVs.
- [`scripts/train_baseline.py`](scripts/train_baseline.py): the classical baseline.
- [`scripts/train_classifier.py`](scripts/train_classifier.py): the deep classifiers.
- [`scripts/run_phase1.sh`](scripts/run_phase1.sh): split files, then baseline.
- [`scripts/run_phase2.sh`](scripts/run_phase2.sh): installs `timm` and `grad-cam`, then trains the three models in turn. It stops with a clear message if `data/splits/` is missing.

All scripts run from a repo clone without `pip install`.

### 3.13 Exploratory Data Analysis (EDA) Notebook
- **File**: [`notebooks/01_rscd_eda.ipynb`](notebooks/01_rscd_eda.ipynb), satisfying Requirement **C1**.
- Class counts per split, a sample grid per friction class, image sizes (PIL reports width x height), brightness and saturation per class (looking for lighting shortcuts), split sanity checks, and the image pairs flagged by the leak report.
- Set `RSCD_ROOT` to use real data. Without it the notebook builds a mock dataset so every cell still runs.

---

## 4. Step-by-Step Manual Verification Playbook

Run everything from the repository root. One-time setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"      # Phase 1 only needs ".[dev,classical]"
```

### Phase 1: Data Pipeline, Metrics and Classical Baseline

#### Test 1: Run the Complete 67-Test Suite
```bash
pytest -v
```
**What to look for** (counts per file):

| File | Tests | Covers |
| :--- | :--- | :--- |
| `tests/test_setup.py` | 4 | package import, config loading, friction class order, damage class ids |
| `tests/test_metrics.py` | 14 | hand-computed confusion matrix, macro-F1, missing classes, bad labels, scikit-learn parity |
| `tests/test_rscd.py` | 19 | label mapping, both folder layouts, balanced and seeded subset, near-duplicate index vs brute force, planted leaks |
| `tests/test_baseline.py` | 2 | feature vector length and end-to-end training |
| `tests/test_dl.py` | 28 | Phase 2 (see Tests 7-13) |

**Success criteria:** `67 passed` in roughly 15 seconds on a laptop CPU. Tests that need scikit-learn or torch are skipped automatically if those are not installed.

#### Test 2: Verify the Configuration Single Source of Truth
```bash
python3 -c "from roadcond.config import load_config; cfg = load_config(); print('Classes:', cfg['rscd']['classes'])"
```
Expected: `Classes: ['dry', 'wet', 'water', 'fresh_snow', 'melted_snow', 'ice']`

#### Test 3: Audit the Anti-Leak Duplicate Detector
```bash
# 1. Synthetic dataset with 2 planted leaks
python3 -m roadcond.data.mock --out data/raw/rscd_mock

# 2. Ask for more images than exist, so EVERY train image is checked
python3 scripts/prepare_rscd.py --root data/raw/rscd_mock \
    --splits-dir data/processed/mock_splits --train-size 10000

# 3. Read the audit
cat data/processed/mock_splits/rscd_leak_report.json
```
**Why 10000:** with a small `--train-size` (say 120), only that many images are drawn and checked, so the two planted leaks are usually never sampled and nothing is rejected. Asking for more than exists forces a full check.

**Expected:**
- Warnings that classes have fewer images than requested. This is normal: the mock has only 20 images per snow/ice label.
- In the report: `"train_candidates_checked": 542`, `"train_rejected_near_duplicate": 2`, `"train_rejected_shared_drive": 0`, and two `examples` pairing `train/.../leak_000.jpg` and `leak_001.jpg` with test images at distance 0.
- `rscd_val.csv` and `rscd_test.csv` list every val and test image (108 each at the default mock size) and are never altered.

#### Test 4: Verify Metric Math & Zero-Division Safety
```bash
python3 -c "
import numpy as np
from roadcond.metrics import confusion_matrix, macro_f1

# 3 classes (0: dry, 1: wet, 2: ice). The model misses class 2 completely.
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 0, 1, 1, 0, 1])
print(confusion_matrix(y_true, y_pred, n_classes=3))
print('Macro-F1:', round(macro_f1(y_true, y_pred, 3), 4))
"
```
Expected:
```text
[[2 0 0]
 [0 2 0]
 [1 1 0]]
Macro-F1: 0.5333
```
Per-class F1 is 0.8, 0.8 and 0.0, so the macro average is (0.8 + 0.8 + 0.0) / 3 = 0.5333. The failure on the rare class is fully penalised.

#### Test 5: Run the Classical Baseline on Mock Data
```bash
python3 scripts/train_baseline.py --root data/raw/rscd_mock \
    --splits-dir data/processed/mock_splits --cache-dir data/processed/mock_cache \
    --results artifacts/mock_results/baseline.json --eval-test
```
Expect feature extraction for train, val and test, then a table with `val macro-F1`, `test macro-F1` and accuracy for logistic regression and random forest as mean ± std over seeds 0, 1, 2, plus a per-class precision / recall table. Mock scores near 1.0 are meaningless.

#### Test 6: Verify Code Style
```bash
ruff check src tests scripts
ruff format --check src tests scripts
```
Expected: `All checks passed!`, then `... files already formatted`.

> `ruff check .` also lints the notebook and currently reports 6 style warnings there (mostly line length). They do not affect behaviour.

### Phase 2: Deep Learning Classifier

Phase 2 reads the split CSVs from Phase 1. The commands below use the mock splits from Test 3.

#### Test 7: Verify SmallCNN (Req C3)
```bash
python3 -c "
import torch
from roadcond.config import load_config
from roadcond.models.pretrained import count_parameters
from roadcond.train import build_model
model = build_model('small_cnn', load_config())
out = model(torch.randn(2, 3, 240, 360))
print('output shape:', tuple(out.shape))
print('parameters:', count_parameters(model))
print('stages:', len(model.features))
"
```
Expected:
```text
output shape: (2, 6)
parameters: 390438
stages: 4
```
The automated checks are `test_forward_loss_and_backward[small_cnn]` (forward pass, finite loss, gradients reach every parameter) and `test_small_cnn_is_small_and_has_four_stages`:
```bash
pytest tests/test_dl.py -v -k "small_cnn"
```

#### Test 8: Verify ResNet-50 and EfficientNet-B0 (Req C4)
```bash
python3 -c "
from roadcond.config import load_config
from roadcond.models.pretrained import count_parameters
from roadcond.train import build_model
cfg = load_config()
for name in ('resnet50', 'efficientnet'):
    model = build_model(name, cfg, pretrained=False, freeze_backbone=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(name, '| total', count_parameters(model), '| trainable', count_parameters(model, True), '|', trainable)
"
```
Expected:
```text
resnet50 | total 23520326 | trainable 12294 | ['fc.weight', 'fc.bias']
efficientnet | total 4015234 | trainable 7686 | ['classifier.weight', 'classifier.bias']
```
With the backbone frozen, only the new 6-class head trains (2048 x 6 + 6 = 12,294 for ResNet-50; 1280 x 6 + 6 = 7,686 for EfficientNet-B0). `pretrained=False` skips the weight download; the real runs use `pretrained=True`.

Automated checks:
- `test_freezing_leaves_only_the_head_trainable`: only the head has gradients, then everything after unfreezing.
- `test_freeze_batchnorm_stats_survives_train_mode`: BatchNorm stays in eval mode while frozen.
- `test_frozen_epoch_trains_only_the_head_and_keeps_bn_statistics`: runs a real frozen epoch and checks that the first convolution and BatchNorm statistics did not change while the head did.

```bash
pytest tests/test_dl.py -v -k "freez or frozen or forward"
```

#### Test 9: Train End to End on Mock Data (training loop, early stopping, checkpoints)
```bash
python3 scripts/train_classifier.py --model small_cnn \
    --root data/raw/rscd_mock --splits-dir data/processed/mock_splits \
    --out-dir artifacts/mock_dl --seeds 0 --epochs 2 --workers 0 --device cpu \
    --limit 200 --eval-test --gradcam 2
```
Takes about 30 seconds on a laptop CPU. Expect one log line per epoch (`train`, `val`, `macro-F1`, `acc`, with ` *` marking a new best), then a results table:

```text
| model     | val macro-F1 | test macro-F1 | test accuracy | test ECE before -> after | temperature | best epochs |
```
Check the files it wrote:
```bash
ls artifacts/mock_dl/checkpoints/small_cnn_seed0 artifacts/mock_dl/results
```
```text
best.pt  gradcam_test.png  history.json  last.pt  loss_curves.png  reliability_test.png
small_cnn.json
```
Open the three PNGs. `history.json` has one entry per epoch (train loss, validation loss, macro-F1, accuracy, learning rate, whether the backbone was frozen).

Repeat with `--model resnet50 --no-pretrained` or `--model efficientnet --no-pretrained` to exercise the other two architectures without a download. Drop `--no-pretrained` to use the real ImageNet weights (downloaded once; EfficientNet-B0 on a CPU took about a minute and a half for 2 epochs on 120 images).

**Resume check** (simulates a Colab timeout): rerun the same command with `--epochs 3 --resume`. The log shows `Resumed from .../last.pt at epoch 2` (a 0-based epoch index), trains only epoch 3, and `history.json` then lists epochs `[0, 1, 2]`. Automated: `test_fit_resumes_from_last_checkpoint`, `test_early_stopper_stops_after_patience_and_keeps_best`.

**Whole pipeline check, automated:** `test_cli_end_to_end_on_mock` runs the full command line (2 seeds, test scoring, Grad-CAM) and checks the results JSON and every output file.

#### Test 10: Verify Grad-CAM (Req C8)
```bash
python3 -c "
import torch
from roadcond.config import load_config
from roadcond.explain import gradcam
from roadcond.train import build_model
model = build_model('small_cnn', load_config())
cam = gradcam(model, torch.randn(2, 3, 240, 360))
print('shape', cam.shape, 'min', round(float(cam.min()), 3), 'max', round(float(cam.max()), 3))
"
```
Expected: `shape (2, 240, 360) min 0.0 max 1.0` (one heatmap per image at input resolution, each scaled to [0, 1]).

Then look at the gallery from Test 9: `artifacts/mock_dl/checkpoints/small_cnn_seed0/gradcam_test.png`. The top row shows the original images and the bottom row the heatmaps. Columns titled `correct` show random correct predictions; columns titled `wrong` show the most confident mistakes. Each title reads `true -> predicted (confidence)`. Warm colours (red) mark pixels that pushed the model toward its predicted class. On a *trained* model, check whether those pixels sit on the road surface or on shortcuts such as sky, shadows or vehicle parts (hypothesis H5).

Automated checks: `test_gradcam_shape_range_and_determinism` (all three models, the same input twice gives the same map), `test_gradcam_explains_the_requested_class`, `test_gradcam_works_with_frozen_parameters`, `test_overlay_returns_uint8_image_and_checks_sizes`, `test_select_examples_takes_most_confident_mistakes`.
```bash
pytest tests/test_dl.py -v -k "gradcam or overlay or select_examples"
```

#### Test 11: Verify Temperature Scaling & ECE (Req C9)
A synthetic model that is 70% accurate but extremely confident:
```bash
python3 -c "
import numpy as np, torch, torch.nn.functional as F
from roadcond.calibration import TemperatureScaler, expected_calibration_error
rng = np.random.default_rng(0)
n, k = 4000, 6
labels = rng.integers(0, k, n)
wrong = (labels + rng.integers(1, k, n)) % k
pred = np.where(rng.random(n) < 0.7, labels, wrong)
logits = rng.normal(0, 0.5, (n, k)); logits[np.arange(n), pred] += 6.0
logits_t = torch.tensor(logits, dtype=torch.float32)
scaler = TemperatureScaler().fit(logits_t, torch.tensor(labels))
before = expected_calibration_error(F.softmax(logits_t, 1).numpy(), labels)
after = expected_calibration_error(F.softmax(scaler(logits_t).detach(), 1).numpy(), labels)
print(f'accuracy {np.mean(pred == labels):.3f}  T {scaler.temperature:.2f}  ECE {before:.3f} -> {after:.3f}')
"
```
Expected: `accuracy 0.700  T 2.45  ECE 0.284 -> 0.030`. The model claimed about 99% confidence at 70% accuracy; scaling by T = 2.45 brings the confidence down to match.

Automated checks (`pytest tests/test_dl.py -v -k "ece or temperature or reliability"`):
- `test_ece_is_zero_for_perfect_confident_predictions` and `test_ece_hand_example` (confidence 0.9 at 50% accuracy gives ECE = 0.4).
- `test_reliability_bins_count_every_sample`.
- `test_temperature_scaling_fixes_overconfident_logits`: T > 1.5, NLL drops, predictions are unchanged, ECE at least halves.

In a real run, the numbers are in `results/<model>.json` under `models -> <model> -> [run] -> test -> calibration`: `temperature`, `ece_before`, `ece_after`, `nll_before`, `nll_after`, and the bin data behind `reliability_test.png`. Remember that the mock run in Test 9 can give T < 1 (it gave 0.03): with a tiny, barely trained model the network is under-confident, and the scaler correctly sharpens it. Do not read anything into mock calibration numbers. Scaling can even make ECE *worse* on test: a 1-epoch `--model resnet50 --no-pretrained --limit 60` run went from 0.157 to 0.315, because T was fitted on a tiny validation set. That is exactly the outcome H4 has to be able to fail with, so always compare `ece_before` and `ece_after` on test rather than assuming scaling helps.

#### Test 12: Verify the Dataset & Transforms
```bash
pytest tests/test_dl.py -v -k "transform or dataset or loader"
```
Checks that tensors are `(3, H, W)` floats, labels match the CSV, an unknown class raises an error, evaluation transforms are deterministic, training transforms are reproducible under a fixed seed, and that `strength=0` removes the colour jitter.

#### Test 13: Run the Whole Phase 2 Test File
```bash
pytest tests/test_dl.py -v
```
Expected: `28 passed` in about 15 seconds on a laptop CPU, using `pretrained=False` everywhere (no network).

---

## 5. Reading the Results Against the PRD Hypotheses

Once real results exist (`artifacts/results/baseline.json`, `small_cnn.json`, `resnet50.json`, `efficientnet.json`), each deep-model file has the same layout as the baseline file:

| Hypothesis | Where to look | Counts as confirmed if |
| :--- | :--- | :--- |
| **H1** fine-tuned > small CNN > classical | `test macro-F1` column of each model's printed table (mean ± std over 3 seeds) | the order holds on the untouched test set over 3 seeds |
| **H2** similar classes are confused most | `test -> confusion_matrix` in the JSON (rows true, columns predicted; classes ordered dry to ice) | the largest off-diagonal cells are between neighbouring classes (wet/water, fresh/melted snow) |
| **H3** imbalance hurts rare classes | not testable yet: needs C6 | |
| **H4** over-confident, fixed by temperature scaling | `test -> calibration -> ece_before` vs `ece_after`, and `reliability_test.png` | ECE drops after scaling **on test** |
| **H5** Grad-CAM shows shortcuts | `gradcam_test.png`, wrong columns | a few wrong predictions are explained by where the model looked |

Reporting rules from the PRD: tune on validation only, run `--eval-test` once per final model, report mean and spread over at least 3 seeds, and publish disappointing results too.

---

## 6. Running the Real Experiments on Google Colab

Full instructions and cells are in the [README](README.md). In short:

1. Use a **GPU runtime**; clone the repo; unzip RSCD to local disk (`/content/rscd`), not Drive.
2. Phase 1: `bash scripts/run_phase1.sh /content/rscd`, then commit `data/splits/` (or keep it in the same session).
3. Phase 2: `bash scripts/run_phase2.sh /content/rscd` (validation only), then once `bash scripts/run_phase2.sh /content/rscd --eval-test --gradcam 4`.
4. After a timeout, rerun with `--resume`. Copy `artifacts/` to Drive before the session ends.
5. Do a short dry run first: `python scripts/train_classifier.py --model resnet50 --root /content/rscd --limit 500 --epochs 1`.

---

## 7. Known Limitations & Assumptions

- **No real results yet.** Everything above is verified on mock data. The GPU / mixed-precision path is untested, and real training time per epoch has not been measured.
- **RSCD layout is assumed**, not confirmed: `train/<label>/` folders and flat `vali_20k` / `test_50k` folders with the label in the file name. If the real download differs, change `rscd.split_dirs` or `rscd.label_source` in `configs/default.yaml`. An unknown label stops the run with an error.
- **`240 x 360` orientation**: the PRD does not say whether this is height x width. The config assumes height x width; the EDA notebook prints the real sizes.
- **Near-duplicate threshold** (4 bits) is a starting value. Plain snow or ice images may be flagged even when they are different scenes: inspect the flagged pairs in the notebook. If file names carry a drive or date id, set `rscd.dedup.drive_id_regex` to also catch frames from the same drive.
- **Mock classes differ only by colour**, so mock scores are near-perfect (or erratic on tiny runs) and say nothing about real performance.
- **Not built yet:** imbalance handling (C6), hyperparameter sweep and tracker (C7), ViT/DINOv2 (C10), all detection work and the demo.
- The EDA notebook has 6 `ruff` style warnings; `src`, `tests` and `scripts` are clean.

---

## 8. What Remains to Be Built (Next Phases)

```mermaid
flowchart TD
    Done["Done: Phases 0-2<br/>(scaffolding, RSCD pipeline, baseline C1-C2-C5,<br/>SmallCNN C3, pretrained C4, Grad-CAM C8, calibration C9)"]
    Real["Next: real runs on Colab GPU<br/>(check data layout, train 3 models x 3 seeds, fill H1-H5)"]
    Extra["Remaining classifier items<br/>(imbalance C6, sweep C7, optional C10)"]
    Phase3["Phase 3: Cross-country damage detection<br/>(Ultralytics YOLO on RDD2022: Japan -> Norway/Czech/USA, D1-D5,<br/>Albumentations for box-aware augmentation)"]
    Phase4["Phase 4: ONNX export & serving<br/>(FastAPI endpoint with limits note, W3)"]
    Phase5["Phase 5: Survey-sheet frontend<br/>(Vite + React + Tailwind + D3, W1-W2)"]

    Done --> Real
    Real --> Extra
    Real --> Phase3
    Extra --> Phase4
    Phase3 --> Phase4
    Phase4 --> Phase5
```

### Immediate next step
Run the real experiments (section 6): confirm the RSCD layout with the EDA notebook, build the real split files, then train the three classifiers and read the results against H1-H5.
