# Cross-country domain shift in road damage detection

*Write-up of the RDD2022 study (PRD R2). Draft: the method is final; the results block fills itself
from the saved result files, and section 5 is written after the real runs. Nothing below reports a
number that has not been measured.*

## 1. Question

A road authority that buys a damage detector trained on one country's roads will use it on
another's. How much accuracy is lost, and what is the cheapest way to win it back?

I test three hypotheses from the project plan (PRD H6 to H8):

- **H6.** A YOLO detector trained on Japan loses accuracy on Norway, the Czech Republic and the USA.
- **H7.** A small number of labelled target images recovers a large part of that gap.
- **H8.** Fine-tuning with labels beats augmentation alone, and pseudo-labelling lands in between.

H6 is the "domain shift" claim. H7 and H8 compare ways to repair it. Either outcome is a finding: a
refuted hypothesis with a good explanation counts.

## 2. Data

RDD2022 has road images from six countries with four damage classes: longitudinal crack (D00),
transverse crack (D10), alligator crack (D20) and pothole (D40). Japan is the **source** (13,133
images); Norway, the Czech Republic and the USA are the **targets**. India and the two China sets
are left out to keep the study to three targets.

**The official test labels are not public**, so there is no official held-out set to score on. Each
country gets its own seeded 70 / 10 / 20 % split of its labelled images into a training pool,
validation and test split. Pool images that nearly duplicate a validation or test image (a 64-bit
difference hash within 4 bits) are removed, because consecutive frames of one drive would otherwise
leak across the split. The exact file lists are committed (`data/splits/rdd_<country>_*.txt`), so
every number is reproducible. The test split of each country is scored once per final model.

Consequence for readers: **these scores are on my own split, not on the RDD2022 challenge test set,
and cannot be compared with published leaderboard numbers.** The 2022 challenge winner's 76.9 % F1
(six countries, official test) is a reference point only.

## 3. Method

**Detector.** YOLOv8n (Ultralytics), 640 px, 30 epochs, trained on at most 5,000 Japanese images
(a cap that lets the run fit a free Colab session). This is a small model on purpose: the aim is a
fair comparison of recovery methods, not a state-of-the-art detector.

**Metrics** (PRD): mAP at IoU 0.5 (headline) and 0.5 to 0.95 from Ultralytics, plus F1 at a fixed
confidence of 0.25 from my own matcher. The **relative gap** is
`(source mAP - target mAP) / source mAP`. The **share of gap recovered** by a method is
`(method mAP - zero-shot mAP) / (source mAP - zero-shot mAP)`; it is reported as "n/a" when the
target already scores at least as well as the source, because nothing is then lost to recover.

**Recovery methods**, all starting from the same source detector:

1. **Stronger augmentation (D2).** Retrain on Japan only, with weather-style, pixel-only
   augmentation (snow glare, fog, darkening, shadows, sun flare, contrast changes; boxes are not
   moved) and no target images.
2. **Fine-tuning with labels (D3, D5).** Fine-tune on 10, 50 or 200 labelled target images, with
   the backbone frozen and a *fixed* recipe (AdamW, learning rate 0.001, the final weights are
   scored). The recipe is never tuned on target data. Budgets are nested within a seed (the 10
   images are inside the 50, inside the 200), over **three seeds**, so the learning curve is not
   confounded by different image draws.
3. **Pseudo-labelling (D4).** Label unlabelled target images with the source detector, keep boxes
   above a confidence threshold, retrain. The threshold is chosen from {0.3, 0.5, 0.7} on target
   *validation* images. This is the one place where target labels influence a "label-free"
   method, and it is stated in the results file.

All fine-tuned numbers are mean ± sample standard deviation over seeds; the source detector is a
single training run, so its numbers have no spread.

## 4. Results

mAP@0.5 on each target country's held-out test split. Regenerate with
`python scripts/render_results.py --results-dir artifacts/results --section detection --readme docs/study_writeup.md`.

<!-- results:start -->

#### Cross-country damage detection (YOLOv8n, RDD2022)

Pending: the detection experiments have not been run (scripts/run_phase3.sh).

<!-- results:end -->

Plots (`artifacts/.../learning_curve.png`) and the full tables
(`artifacts/results/detection_report.md`) are written by `python scripts/train_detection.py report`;
the demo's **Study** tab draws the same measured numbers from `study.json`.

## 5. Findings

*To be written after the real runs. Decision rules, fixed in advance from the project plan:*

| Hypothesis | Counts as confirmed if | Outcome |
|---|---|---|
| H6 | target mAP is lower than source mAP, with the gap stated per country | pending |
| H7 | the learning curve rises over 0, 10, 50, 200 labels, and I can say where it flattens | pending |
| H8 | the ranking fine-tuning > pseudo-labels > augmentation holds across seeds, or I report that it does not | pending |

For each country the findings paragraph will state: the zero-shot gap; the label budget at which
half the gap is recovered (or that it is never reached); whether the spread across seeds is small
enough to separate the methods at equal budget; and, where the order is not clear, say so rather than
rank on means alone.

## 6. Threats to validity

- **Own split, not the official test set** (section 2). Scores are internally consistent, not
  externally comparable.
- **Small detector, capped training set, one source run.** The source mAP has no seed spread, so a
  small gap may be noise. Differences smaller than the seed standard deviation should not be read as
  effects.
- **Countries differ in more than "country".** Camera, vehicle, image size, season and annotation
  habits vary between sets, so the gap measures all of them together, not roads alone.
- **Few-shot noise.** With 10 labelled images the result depends heavily on which 10; the three
  seeds only partly cover that.
- **Near-duplicate removal is approximate.** A 64-bit difference hash catches repeated frames, not
  every near-identical scene.
- **Licence.** Ultralytics is AGPL-3.0; decide the repository's licence before making it public.

## 7. Reproduce

```bash
pip install -e ".[dev,detect]"
bash scripts/run_phase3.sh <path to RDD2022>      # splits, source, augmentation, few-shot, pseudo, report
python scripts/render_results.py --results-dir artifacts/results --section detection --readme docs/study_writeup.md
```

Seeds, split files, recipe and thresholds are in `configs/default.yaml` and `data/splits/`.
