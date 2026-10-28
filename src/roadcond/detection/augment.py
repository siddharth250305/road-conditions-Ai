"""Weather augmentation for the source detector (PRD D2).

The question D2 asks: how much of the domain gap closes with augmentation alone, using no
target images at all? Norwegian roads are often wet, snowy or low-contrast, so the pipeline
imitates snow glare, wet-pavement look and contrast changes on the Japanese training images.

Every transform is pixel-level, so boxes never move. Rain streaks are deliberately left out:
long thin streaks look like longitudinal cracks and would teach the detector wrong cues.
Ultralytics saves its augmentations in checkpoints, so only built-in Albumentations classes
are used (no A.Lambda).
"""

import warnings
from collections.abc import Sequence

import numpy as np


def weather_transforms(strength: float = 1.0) -> list:
    """Albumentations transforms for ultralytics `model.train(augmentations=...)`.

    strength scales both how often and how strongly they apply; 0 returns no transforms.
    """
    import albumentations as A

    if strength <= 0:
        return []
    s = float(strength)
    p = min(1.0, 0.5 * s)

    def mild(value: float, cap: float) -> float:
        return min(cap, value * s)

    return [
        A.OneOf(  # snow glare: bleached, hazy, over-exposed
            [
                A.RandomSnow(snow_point_range=(0.05, 0.05 + mild(0.2, 0.45)), brightness_coeff=2.0),
                A.RandomFog(fog_coef_range=(0.1, 0.1 + mild(0.3, 0.6)), alpha_coef=0.1),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.1 + mild(0.3, 0.5)),
                    contrast_limit=(-mild(0.4, 0.6), -0.05),
                ),
            ],
            p=p,
        ),
        A.OneOf(  # wet pavement: darker, shadowed, specular highlights
            [
                A.RandomGamma(gamma_limit=(max(40, 100 - mild(60, 60)), 100)),
                A.RandomShadow(
                    shadow_roi=(0, 0.3, 1, 1), shadow_intensity_range=(0.3, 0.3 + mild(0.3, 0.4))
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0.3, 1, 1),
                    src_radius=int(60 + 60 * min(s, 2)),
                    num_flare_circles_range=(2, 4),
                ),
            ],
            p=p,
        ),
        A.OneOf(  # contrast changes
            [
                A.CLAHE(clip_limit=(1.0, 1.0 + mild(3.0, 6.0))),
                A.RandomBrightnessContrast(
                    brightness_limit=mild(0.2, 0.4), contrast_limit=mild(0.4, 0.7)
                ),
            ],
            p=p,
        ),
    ]


def apply_to_sample(
    image: np.ndarray,
    yolo_rows: Sequence[Sequence[float]],
    strength: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the pipeline on one image with YOLO boxes (rows of cls cx cy w h): (image, rows).

    For tests and previews; training applies the same transforms inside Ultralytics.
    """
    import albumentations as A

    rows = np.asarray(yolo_rows, dtype=float).reshape(-1, 5)
    with warnings.catch_warnings():  # "no transform to process bboxes": all ours are pixel-level
        warnings.simplefilter("ignore", UserWarning)
        pipeline = A.Compose(
            weather_transforms(strength),
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
            seed=seed,
        )
    out = pipeline(image=image, bboxes=rows[:, 1:].tolist(), labels=rows[:, 0].astype(int).tolist())
    boxes = np.asarray(out["bboxes"], dtype=float).reshape(-1, 4)
    labels = np.asarray(out["labels"], dtype=float).reshape(-1, 1)
    return out["image"], np.hstack([labels, boxes])
