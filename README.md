# Road Condition AI

Camera-based **road surface (friction) classification** on RSCD, plus a **cross-country road-damage
detection study** on RDD2022, served as a small web demo. It is a student portfolio project: the
code, the experiments and the demo are real, and every number below comes from saved result files
(or says "pending").

**Demo status:** the web app, API, ONNX export and Docker image are built and tested on mock data.
The real training runs (Colab or Kaggle GPU) have not been added yet, so the demo ships with a mock
model and says so on every screen.

## Architecture

```mermaid
flowchart LR
  subgraph R["Research: Colab / Kaggle GPU"]
    RSCD[(RSCD)] --> CLS["Phases 1-2<br/>baseline, SmallCNN, ResNet-50,<br/>EfficientNet-B0, calibration, Grad-CAM"]
    RDD[(RDD2022)] --> DET["Phase 3<br/>YOLOv8n source detector,<br/>domain shift, 3 recovery methods"]
  end
  CLS -- "best.pt + temperature T" --> EXP["scripts/export_model.py"]
  DET -- "result JSON" --> EXP
  EXP --> BUNDLE[("serving/<br/>model.onnx, meta.json, study.json")]
  subgraph D["Demo: one Docker image, port 7860"]
    BUNDLE --> API["FastAPI + ONNX Runtime (CPU)<br/>/predict /study /health"]
    API --> UI["React app<br/>3D crossroads home, report, study, limits"]
  end
```

Research and serving meet only in the `serving/` folder. The server needs no PyTorch: the ONNX graph
contains the normalisation and a class-activation-map output, so heatmaps need no gradients.

## Quick start

```bash
make setup        # one command: Python and frontend dependencies (Python 3.11+, Node 22+)
make demo-mock    # one command: builds a random-weight demo bundle and serves http://localhost:8000
```

`make demo-mock` needs no download and no training; its answers mean nothing and are labelled so.
For the real model, train on Colab (Phases 1-3 below), then:

```bash
python scripts/export_model.py bundle --checkpoint <best.pt> --results <results/model.json> --out serving
python scripts/export_model.py study --results-dir artifacts/results --out serving/study.json
make demo                  # or: docker compose up --build   (http://localhost:7860)
```

Other targets: `make test` (ruff, pytest, frontend type check and tests: what CI runs), `make docker`.

## Results

Generated from the saved result files by `python scripts/render_results.py --results-dir artifacts/results --readme README.md`.
Cells say "pending" until the real runs exist; nothing here is typed in by hand.

<!-- results:start -->

#### Surface classification (RSCD, six friction classes)

| model         | val macro-F1 | test macro-F1 | test accuracy | test ECE before -> after |
|---------------|--------------|---------------|---------------|--------------------------|
| logreg        | pending      | pending       | pending       | pending                  |
| random_forest | pending      | pending       | pending       | pending                  |
| small_cnn     | pending      | pending       | pending       | pending                  |
| resnet50      | pending      | pending       | pending       | pending                  |
| efficientnet  | pending      | pending       | pending       | pending                  |

No classifier results found yet: run the Phase 1 and Phase 2 scripts.

#### Cross-country damage detection (YOLOv8n, RDD2022)

Pending: the detection experiments have not been run (scripts/run_phase3.sh).

<!-- results:end -->

Plots are written next to the results by the training scripts (`loss_curves.png`, `reliability_<split>.png`,
`gradcam_<split>.png`, the detection learning curve); the write-up is in
[`docs/study_writeup.md`](docs/study_writeup.md).

## Limitations and future work

- **No real results yet.** Everything above was built and tested on synthetic mock data; the demo's
  model is random weights and is labelled as such. The tables fill in after the Colab / Kaggle runs.
- **RSCD comes from China only.** The classifier's behaviour on Nordic roads, at night or with rain on
  the lens is unknown. The demo shows this on every screen and returns "Not sure" below 50 %
  calibrated confidence (a default I chose; the PRD gives no value, so tune it on validation data).
- **The demo cannot tell that a photo is not a road.** The model has six road classes and no
  "not a road" class; low confidence is the only guard.
- **Own detection split.** RDD2022 test labels are not public, so detection scores are on my own
  per-country split and are not comparable with the challenge leaderboard.
- **The heatmap is a class activation map** at the last feature map (equal to Grad-CAM there, and
  checked by a test): coarse, and evidence of where the model looked, not of why it is right.
- **Licences.** Ultralytics is AGPL-3.0; check dataset terms and choose the repository licence before
  publishing code, weights or sample images. The bundled sample pictures are generated placeholders.
- **Future work:** class-imbalance comparison and a hyperparameter sweep (C6, C7), a ViT / DINOv2
  backbone (C10), radar data for road surface condition (no public dataset found), and visual
  localisation under changing conditions (L1).

## Install options

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

## Phase 5: web app, Docker and CI (W1, W2, R1, R2)

A React app built from a hero design over a dusk drone photo of a city crossroads. The home screen
is that photo laid flat as the ground of a 3D scene (three.js through React Three Fiber, loaded
lazily), with surface layers fitted over its roads: the Dry, Wet, Snow and Ice chips repaint the
roads in place. A dark site bar links Home, Demo, How it works, Study and Model, with GitHub and
Get started on the right. On Report the photo sits in
a frame with give-way corner marks, a friction ruler runs dry to ice, and the verdict is painted on.
Vite, React, TypeScript and Tailwind CSS; D3 does the chart maths and React draws the SVG. Fonts
(Barlow and Barlow Condensed, drawn from California highway signs) are bundled with `@fontsource`, so the page makes no third-party requests.
The visual system is in [`DESIGN.md`](DESIGN.md).

| Screen / state | What it shows |
|---|---|
| Home | hero over the drone photo with a 3D road-surface layer (Dry, Wet, Snow, Ice chips), Try the Demo and View on GitHub, a How it works section; the plain photo without WebGL |
| Report, empty | a blank photo frame as the drop area, three generated sample pictures |
| Report, result | photo with a 180 ms heatmap crossfade toggle, ruler marker (300 ms ease-out), painted-on verdict (160 ms), calibrated confidence |
| Report, low confidence | "Not sure", the top two classes, never a forced label |
| Report, error / offline / no model | fixed messages; retry only where retrying can help |
| Study | slope chart (source mAP to target mAP per country) and learning curve over 0, 10, 50, 200 labels, with the numbers in a table |
| Limits | a strip on every screen and an expandable panel: training data, country, what the model has not seen, calibration, provenance |

```bash
cd frontend && npm ci
npm run dev            # http://localhost:5173, proxies the API on :8000 (run `make demo-mock` in another shell)
npm test               # 87 tests: API client, state machine, ruler, charts, design rules, home screen, flows
npm run build          # writes frontend/dist, which the API serves at /
```

- **One service.** `api/main.py` serves `frontend/dist` at `/` after every API route (set
  `ROADCOND_FRONTEND_DIR` to change the folder), so the page and the API share one origin.
- **Docker.** `docker compose up --build` builds the React page with Node, then a slim Python image
  (no torch) with FastAPI and ONNX Runtime on port 7860, as a non-root user. The model bundle is read
  from `./serving` (not in git); without one the page loads and says no model is loaded.
- **Hugging Face Space (Docker SDK).** The image already follows the Space rules (port 7860, uid 1000).
  Not deployed yet. In the Space repository, put this at the top of its `README.md`, add the
  `serving/` files with Git LFS (ResNet-50's `model.onnx` is about 94 MB), and push:

  ```yaml
  ---
  title: Road Condition AI
  sdk: docker
  app_port: 7860
  ---
  ```
- **Sample pictures** are drawn by `python scripts/make_samples.py` (no licence terms, no class
  label). Replace `frontend/public/samples/sample-N.jpg` with photos you may use before publishing.
- **Accessibility.** Keyboard use throughout, a yellow focus ring on ink, alt text stating class and
  confidence, results announced through a live region, all motion off under `prefers-reduced-motion`,
  and a dark variant. A test checks the contrast of every text pair against the stylesheet.
- **CI.** `.github/workflows/ci.yml` runs ruff and pytest on Python 3.11 and 3.12, the ML and ONNX
  tests with CPU torch, the frontend type check, tests and build, and a Docker build with a smoke test.

Local check without any download or training: `make test`.

## Review routing

The classifier gives one label per image. The review router decides how much **human** review that
label needs, and writes down why, so every decision can be audited later. Each image gets exactly
one mode. There is no automatic-accept mode: `light_review` is the lightest outcome.

| Mode | Meaning |
|---|---|
| `light_review` | a person glances at the model's label |
| `full_review` | a person checks the image carefully |
| `second_review` | two people check it independently |

```mermaid
flowchart LR
  P[("baseline_probs.npz<br/>per-model probabilities")] --> C["Case<br/>(image id, probs, true label)"]
  C -- "CaseView: no label, no file name" --> S1["CalibratedConfidenceProvider<br/>top-1 after temperature T"]
  C -- "CaseView" --> S2["AgreementProvider<br/>do the models agree?"]
  S1 --> R{{"ReviewRouter"}}
  S2 --> R
  R -- "signals (None if a provider failed)" --> POL["ThresholdPolicy<br/>safety class, agreement,<br/>conf < low / high"]
  POL -- "mode + reasons" --> R
  R --> LOG[("decisions.jsonl<br/>append-only, + outcome events")]
```

- **Modular.** `src/roadcond/routing/` uses only the standard library and NumPy. Providers implement
  `ConfidenceProvider` and policies implement `ReviewPolicy`. The router receives them through its
  constructor and knows no concrete class, so a new signal or policy needs no router change.
- **Rules** (`ThresholdPolicy`, strictest rule wins). Any model predicting a safety-critical class
  (`ice`) or models disagreeing gives `second_review`. Confidence below `low` (0.50) gives
  `second_review`, below `high` (0.85) gives `full_review`, otherwise `light_review`. A missing signal
  or an unusable prediction is never lighter than `full_review`. All numbers are in the `routing:`
  section of `configs/default.yaml`.
- **Failures only make review stricter.** A provider that raises or returns NaN, inf or a value
  outside [0, 1] becomes a missing signal, and the error is logged. A policy that raises gives
  `full_review` with `fallback:policy_error`. Providers and policies never see the true label or
  the image's file name (RSCD file names contain the label).
- **Audit log.** One JSON line per decision with schema version, uuid, UTC timestamp, predictions,
  signals, provider errors, policy name and version, a sha256 of the routing config, mode, reasons
  and a fallback flag. `DecisionLog.record_outcome` appends a separate outcome line and never edits
  old ones, and `read_decisions` joins them. Re-running moves the old log aside instead of deleting it.
- **Other models later.** Any model can be routed once its probabilities are saved in the same
  `.npz` format (`roadcond.routing.probs`). The served CNN is routed this way, offline and live:
  see "Live review routing" below.

```bash
# 1. Save first-seed probabilities of the two classical baselines (val and test).
#    Uses the cached features in data/processed when --root matches the earlier run (12.5 min on an 11-core laptop CPU).
python scripts/train_baseline.py --root <RSCD> --eval-test --save-probs
# 2. Fit T on val, route every test image, write artifacts/routing/{decisions.jsonl,summary.json}
make route            # = python scripts/route_images.py --archive-existing
```

### Results: random forest + logistic regression, RSCD test set, seed 0

From `artifacts/routing/summary.json` (policy `threshold` 1.0.0, `high` 0.85, `low` 0.50, safety
class `ice`; 49,500 test images; random forest is the primary model).

| Mode | Images | Share | Random-forest error rate in mode |
|---|---:|---:|---:|
| `light_review` | 8,693 | 17.6% | 10.9% (950 / 8,693) |
| `full_review` | 13,416 | 27.1% | 27.1% (3,632 / 13,416) |
| `second_review` | 27,391 | 55.3% | 50.2% (13,752 / 27,391) |

| Measure | Value |
|---|---:|
| Random-forest accuracy (test) | 63.0% |
| Temperature (fitted on val) | 0.478 |
| ECE on test, before / after scaling (15 bins) | 0.182 / 0.044 |
| ECE on val, before / after scaling | 0.196 / 0.050 |
| Error catch rate (wrong predictions sent to full or second review) | 94.8% (17,384 / 18,334) |
| Share of all images sent to full or second review | 82.4% |
| Lift over random routing of the same volume | 1.15 |
| True-ice images sent to second review | 91.5% (2,151 / 2,350) |
| True-ice images left in light review | 49 |
| Precision of the ice rule (fired / truly ice) | 25.5% (7,425 fired) |
| Fallbacks (provider or policy failures) | 0 |

Threshold sweep over `high` (descriptive only, computed on test; `high` was set in the config
before the run, not chosen from this table):

| `high` | light | full | second | error rate in light |
|---:|---:|---:|---:|---:|
| 0.60 | 36.6% | 8.1% | 55.3% | 18.2% |
| 0.65 | 32.6% | 12.1% | 55.3% | 17.1% |
| 0.70 | 28.7% | 15.9% | 55.3% | 15.9% |
| 0.75 | 24.9% | 19.7% | 55.3% | 14.6% |
| 0.80 | 21.2% | 23.5% | 55.3% | 12.7% |
| 0.85 | 17.6% | 27.1% | 55.3% | 10.9% |
| 0.90 | 13.7% | 31.0% | 55.3% | 9.0% |
| 0.95 | 9.5% | 35.1% | 55.3% | 6.8% |

**How to read this.**

- Mistakes are concentrated in the stricter modes: the error rate rises from 10.9% in light review
  to 50.2% in second review.
- The routing is only a little better than random at finding errors. The 94.8% catch rate is
  earned by sending 82.4% of all images to full or second review, a lift of 1.15 over random
  routing of the same volume.
- With classical baselines at 63% accuracy, most images need a careful look. Second review is
  55.3% of all images and does not move in the sweep, because it comes from the ice rule, model
  disagreement and `low`, not from `high`.
- The ice rule sends 91.5% of true-ice images to second review, but only a quarter of the images
  it flags are truly ice.

**Caveats.**

- The review is simulated: no human outcomes exist yet.
- These are single-seed results without confidence intervals, from the first-seed models only, so
  they do not match the 3-seed means in the baseline tables.
- Temperature scaling of random-forest vote fractions is a heuristic (zero votes are floored at
  1e-12). T < 1 sharpens the forest's spread-out votes. It is fitted on val, which also chose the
  forest's depth.
- Error rates depend on the official val/test class mix, while training was balanced.
- `config_hash` covers the `routing:` section only, including its paths.

### Live review routing (the served CNN)

The API routes every uploaded photo with the same router and policy, once the exported model has
passed the same offline check. Steps:

```bash
# 1. The served model's probabilities on RSCD val and test, through the server's own
#    decoding, resizing and temperature (about 50 min for 69,360 images on an 11-core laptop CPU).
python scripts/save_bundle_probs.py --root <RSCD>
# 2. Offline check (configs/routing_cnn.yaml): fit T on val, route the test set, then write
#    serving/routing.json, which switches routing on in the API for this exact model.
make route-cnn
# 3. Serve. Every /predict answer now carries a `review`; reviewers work the queue at #/review.
ROADCOND_REVIEWER_TOKENS="alex:<16+ chars>,sam:<16+ chars>" ROADCOND_REVIEW_IMAGE_DIR=logs/photos make demo
# 4. Later: compare the reviewers' labels with the offline check.
make review-report
```

- **One model, same rules.** The API serves one model, so agreement is off; `ice`, `low` 0.50 and
  `high` 0.85 are the baseline's settings, fixed before any CNN result was seen.
- **Tied to the model.** `routing.json` carries the model's `onnx_sha256`, the routing config's
  hash and the temperature fitted on val. If the served model changes, routing switches off (and
  `/health` says why) until the check is rerun.
- **Reviews.** Light and full reviews finish after one reviewer; a second review when two
  reviewers agree. Reviewers sign in with a token from `ROADCOND_REVIEWER_TOKENS`, cannot review a
  photo twice and never see another reviewer's label. Every decision and review is an appended
  line in `ROADCOND_DECISION_LOG` (`logs/decisions.jsonl`; `/data/decisions.jsonl` in Docker).
  Photos are kept for reviewers only when `ROADCOND_REVIEW_IMAGE_DIR` is set.
- **Monitoring.** `make review-report` gives the model's error rate per mode against the reviewers'
  labels, with a 95% interval, and flags when light reviews go wrong more often than the offline
  check below said (once at least 30 are finished).

#### Results: ResNet-50 (served ONNX bundle), RSCD test set, seed not recorded

From `artifacts/routing_cnn/summary.json` (policy `threshold` 1.0.0, `high` 0.85, `low` 0.50,
safety class `ice`; 49,500 test images). The bundle's `meta.json` does not record the
training seed.

| Mode | Images | Share | ResNet-50 error rate in mode |
|---|---:|---:|---:|
| `light_review` | 27,282 | 55.1% | 3.6% (990 / 27,282) |
| `full_review` | 17,010 | 34.4% | 29.2% (4,973 / 17,010) |
| `second_review` | 5,208 | 10.5% | 36.2% (1,884 / 5,208) |

| Measure | Value |
|---|---:|
| ResNet-50 accuracy (test) | 84.1% |
| Temperature (fitted on val) | 0.987 |
| ECE on test, before / after scaling (15 bins) | 0.0063 / 0.0044 |
| ECE on val, before / after scaling | 0.0048 / 0.0045 |
| Error catch rate (wrong predictions sent to full or second review) | 87.4% (6,857 / 7,847) |
| Share of all images sent to full or second review | 44.9% |
| Lift over random routing of the same volume | 1.95 |
| True-ice images sent to second review | 96.5% (2,268 / 2,350) |
| True-ice images left in light review | 29 |
| Precision of the ice rule (fired / truly ice) | 78.5% (2,878 fired) |
| Fallbacks (provider or policy failures) | 0 |

Threshold sweep over `high` (descriptive only, computed on test; `high` was set before the run):

| `high` | light | full | second | error rate in light |
|---:|---:|---:|---:|---:|
| 0.60 | 80.8% | 8.7% | 10.5% | 10.3% |
| 0.65 | 76.3% | 13.2% | 10.5% | 8.8% |
| 0.70 | 71.7% | 17.8% | 10.5% | 7.6% |
| 0.75 | 67.0% | 22.5% | 10.5% | 6.3% |
| 0.80 | 61.6% | 27.9% | 10.5% | 5.0% |
| 0.85 | 55.1% | 34.4% | 10.5% | 3.6% |
| 0.90 | 47.5% | 42.0% | 10.5% | 2.5% |
| 0.95 | 36.9% | 52.5% | 10.5% | 1.3% |

**How to read this.**

- The CNN is far better suited to routing than the classical baselines: over half the photos
  (55.1%) need only a glance, and the model is wrong on
  3.6% of them, against 10.9% for the random forest's light review.
- 87.4% of the model's mistakes reach a careful review while only
  44.9% of photos get one, a lift of 1.95 over random routing.
- The served probabilities were already close to calibrated (test ECE 0.0063),
  so T = 0.987 changes almost nothing; the page's confidence and the review
  confidence therefore agree to within about a point.
- The ice rule now catches 96.5% of true ice with 78.5% precision.

**Caveats.** One model from one training run, no confidence intervals; error rates use RSCD's
labels and the official val/test class mix. The live numbers will come from reviewers' labels, not
RSCD's, and only `make review-report` on a real log can say whether they hold. The bundle's
`meta.json` says temperature 1.0 with no calibration split; T above was fitted for routing only.
