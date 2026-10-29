"""Serving logic for the demo API (PRD W1, W2): ONNX Runtime on CPU, no torch.

Reads a bundle written by roadcond.export (model.onnx + meta.json, optionally study.json) and
answers one question: what is this road photo? Every answer carries the calibrated confidence,
the top two classes, a "Not sure" flag instead of a forced label when confidence is low, a
heatmap of where the model looked, and a note on what the model has and has not seen.

Uploads are untrusted: the file type is decided by decoding, never by the client's claims; the
size and pixel count are limited before decoding; nothing is written to disk or logged.
"""

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)

SUPPORTED_FORMAT_VERSION = 1
UNSURE_LABEL = "Not sure"


class ModelUnavailableError(RuntimeError):
    """The bundle is missing, broken or from an unsupported version."""


class InvalidImageError(ValueError):
    """The upload is empty, not an image, or corrupt."""


class ImageTooLargeError(InvalidImageError):
    """The upload has too many bytes or pixels."""


@dataclass(frozen=True)
class ServingSettings:
    """Limits and display options; env vars override the defaults (see from_env)."""

    uncertainty_threshold: float | None = None  # None: use the bundle's value
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 50_000_000
    heatmap_alpha: float = 0.45

    @classmethod
    def from_env(cls) -> "ServingSettings":
        env = os.environ
        threshold = env.get("ROADCOND_UNCERTAINTY_THRESHOLD")
        return cls(
            uncertainty_threshold=float(threshold) if threshold else None,
            max_upload_bytes=int(float(env.get("ROADCOND_MAX_UPLOAD_MB", "10")) * 1024 * 1024),
            max_pixels=int(env.get("ROADCOND_MAX_PIXELS", "50000000")),
            heatmap_alpha=float(env.get("ROADCOND_HEATMAP_ALPHA", "0.45")),
        )


# --- images ---------------------------------------------------------------------------------


def decode_image(data: bytes, max_pixels: int = 50_000_000) -> Image.Image:
    """Decode an upload to an RGB image, or raise InvalidImageError / ImageTooLargeError."""
    if not data:
        raise InvalidImageError("The uploaded file is empty")
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size  # reads the header only
    except Image.DecompressionBombError:
        raise ImageTooLargeError("The image has too many pixels") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise InvalidImageError("The file is not a readable image") from None
    if width * height > max_pixels:
        raise ImageTooLargeError(
            f"The image has {width * height:,} pixels; the limit is {max_pixels:,}"
        )
    try:
        image.load()
        image = ImageOps.exif_transpose(image)  # phone photos carry their rotation in EXIF
        return image.convert("RGB")
    except Image.DecompressionBombError:
        raise ImageTooLargeError("The image has too many pixels") from None
    except (OSError, ValueError, SyntaxError):
        raise InvalidImageError("The image is corrupt or truncated") from None


def preprocess(image: Image.Image, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(model input (1, 3, H, W) float32 in [0, 1], the resized RGB uint8 image (H, W, 3)).

    Plain Pillow bilinear resizing, so the server needs no torch. It matches the training
    pipeline to within about one grey level; the mean/std normalisation is inside the ONNX graph.
    """
    resized = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
    batch = np.ascontiguousarray(resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
    return batch, resized


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _minmax(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return a / (1e-7 + a.max())


def cam_to_heatmap(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    """One class activation map (h, w) -> heatmap (H, W) in [0, 1].

    Same steps as pytorch-grad-cam: ReLU, scale to [0, 1], bilinear upsampling, scale again.
    """
    small = _minmax(np.maximum(cam, 0).astype(np.float32))
    up = np.asarray(Image.fromarray(small).resize((width, height), Image.Resampling.BILINEAR))
    return _minmax(up)


def _jet(v: np.ndarray) -> np.ndarray:
    """The classic blue-to-red colour map for values in [0, 1]: array (..., 3) in [0, 1]."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def render_overlay(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> bytes:
    """PNG bytes of the picture with the heatmap blended on top."""
    if rgb.shape[:2] != heatmap.shape:
        raise ValueError(f"Image {rgb.shape[:2]} and heatmap {heatmap.shape} sizes differ")
    blended = (1 - alpha) * rgb.astype(np.float32) + alpha * 255.0 * _jet(heatmap)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


# --- the model ------------------------------------------------------------------------------


def limits_note(meta: dict[str, Any]) -> str:
    """What the model was trained on and what it has not seen (PRD W2: shown on every result)."""
    classes = ", ".join(c.replace("_", " ") for c in meta["classes"])
    cal = meta["calibration"]
    ece = (
        f" (expected calibration error {cal['ece_after']:.3f} on the {cal['split']} split)"
        if cal.get("ece_after") is not None
        else ""
    )
    note = (
        f"This model was trained and evaluated only on RSCD: road-surface patches from vehicle "
        f"cameras in China, labelled with six friction levels ({classes}). How well it works on "
        f"roads in other countries, at night, with rain on the lens, or on surfaces outside "
        f"these classes is unknown. The confidence is calibrated on held-out RSCD data "
        f"(temperature {cal['temperature']:.2f}){ece}, so it can be over-confident on other "
        f"roads. The heatmap shows where the model looked, not proof that it is right. "
        f"This is a student project, not a safety system."
    )
    if meta.get("mock"):
        note = "DEMO MODEL BUILT FROM MOCK DATA: its answers mean nothing. " + note
    return note


class Predictor:
    """An ONNX Runtime session plus everything needed to turn logits into an answer."""

    def __init__(self, bundle_dir: str | Path, settings: ServingSettings | None = None):
        import onnxruntime as ort

        self.settings = settings or ServingSettings()
        bundle = Path(bundle_dir)
        for name in ("meta.json", "model.onnx"):
            if not (bundle / name).is_file():
                raise ModelUnavailableError(f"{bundle / name} not found: export a model first")
        try:
            self.meta = json.loads((bundle / "meta.json").read_text())
        except json.JSONDecodeError as err:
            raise ModelUnavailableError(f"meta.json is not valid JSON: {err}") from None
        if self.meta.get("format_version") != SUPPORTED_FORMAT_VERSION:
            raise ModelUnavailableError(
                f"Unsupported bundle format {self.meta.get('format_version')!r}; "
                f"this server reads version {SUPPORTED_FORMAT_VERSION}"
            )
        self.session = ort.InferenceSession(
            str(bundle / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        info = self.meta["input"]
        self.height, self.width = info["height"], info["width"]
        self.input_name = info["name"]
        shape = self.session.get_inputs()[0].shape
        if self.session.get_inputs()[0].name != self.input_name or shape[1:] != [
            3,
            self.height,
            self.width,
        ]:
            raise ModelUnavailableError(f"model.onnx input {shape} does not match meta.json")
        self.classes: list[str] = list(self.meta["classes"])
        self.temperature = float(self.meta["calibration"]["temperature"])
        threshold = self.settings.uncertainty_threshold
        self.threshold = float(
            threshold if threshold is not None else self.meta["serving"]["uncertainty_threshold"]
        )
        self.note = limits_note(self.meta)

    def predict(self, data: bytes, heatmap: bool = True) -> dict[str, Any]:
        """Classify one uploaded image (raw bytes)."""
        image = decode_image(data, self.settings.max_pixels)
        batch, rgb = preprocess(image, self.height, self.width)
        logits, cam = self.session.run(None, {self.input_name: batch})
        probs = softmax(logits[0].astype(np.float64) / self.temperature)
        order = np.argsort(-probs)
        top = int(order[0])
        confidence = float(probs[top])
        uncertain = confidence < self.threshold
        overlay = None
        if heatmap:
            overlay = base64.b64encode(
                render_overlay(
                    rgb,
                    cam_to_heatmap(cam[0, top], self.height, self.width),
                    self.settings.heatmap_alpha,
                )
            ).decode("ascii")
        return {
            "prediction": self.classes[top],
            "confidence": confidence,
            "is_uncertain": bool(uncertain),
            "display_label": UNSURE_LABEL if uncertain else self.classes[top],
            "top2": [
                {"class": self.classes[int(i)], "probability": float(probs[i])} for i in order[:2]
            ],
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs, strict=True)},
            "heatmap_base64": overlay,
            "heatmap_media_type": "image/png",
            "heatmap_class": self.classes[top],
            "limits_note": self.note,
            "temperature": self.temperature,
            "uncertainty_threshold": self.threshold,
            "model": {"name": self.meta["model_name"], "mock": bool(self.meta.get("mock"))},
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.meta["model_name"],
            "classes": self.classes,
            "mock": bool(self.meta.get("mock")),
            "input": {"height": self.height, "width": self.width},
            "uncertainty_threshold": self.threshold,
            "calibration": self.meta["calibration"],
            "training": self.meta.get("training"),
            "provenance": {
                key: self.meta.get("provenance", {}).get(key)
                for key in ("checkpoint_sha256", "onnx_sha256")
            },
        }


def load_study(bundle_dir: str | Path) -> dict[str, Any] | None:
    """The cross-country benchmark JSON of a bundle, or None when it has none."""
    path = Path(bundle_dir) / "study.json"
    if not path.is_file():
        return None
    study = json.loads(path.read_text())
    if study.get("format_version") != SUPPORTED_FORMAT_VERSION:
        raise ValueError(f"Unsupported study format {study.get('format_version')!r}")
    return study
