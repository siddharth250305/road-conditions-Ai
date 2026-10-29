"""Demo API (PRD W1, W2): road photo in, calibrated condition, uncertainty and heatmap out.

    ROADCOND_BUNDLE_DIR=serving uvicorn api.main:app

The model comes from a bundle folder written by `python -m roadcond.export`; research and
serving meet only there. If the bundle is missing the app still starts, /health and /predict
answer 503, and /study keeps working when a study.json is present.

When a built frontend exists (frontend/dist, or $ROADCOND_FRONTEND_DIR) it is served at "/"
after every API route, so the demo is one same-origin service.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from roadcond import __version__
from roadcond.serving import (
    ImageTooLargeError,
    InvalidImageError,
    ModelUnavailableError,
    Predictor,
    ServingSettings,
    load_study,
)

log = logging.getLogger("roadcond.api")

MULTIPART_OVERHEAD = 64 * 1024  # form boundaries and headers around the file bytes
DEFAULT_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"


class ModelInfo(BaseModel):
    name: str
    mock: bool


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    is_uncertain: bool
    display_label: str
    top2: list[dict[str, Any]]
    probabilities: dict[str, float]
    heatmap_base64: str | None
    heatmap_media_type: str
    heatmap_class: str
    limits_note: str
    temperature: float
    uncertainty_threshold: float
    model: ModelInfo


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str | None = None
    classes: list[str] | None = None
    mock: bool | None = None
    detail: str | None = None
    uncertainty_threshold: float | None = None
    calibration: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None


def create_app(
    bundle_dir: str | Path | None = None,
    settings: ServingSettings | None = None,
    frontend_dir: str | Path | None = None,
) -> FastAPI:
    """Build the app around a bundle folder (default: $ROADCOND_BUNDLE_DIR or ./serving).

    frontend_dir: a built frontend to serve at "/" (default: $ROADCOND_FRONTEND_DIR, else
    frontend/dist when it has been built).
    """
    bundle = Path(bundle_dir or os.environ.get("ROADCOND_BUNDLE_DIR", "serving"))
    frontend = Path(frontend_dir or os.environ.get("ROADCOND_FRONTEND_DIR") or DEFAULT_FRONTEND_DIR)
    settings = settings or ServingSettings.from_env()

    predictor: Predictor | None = None
    study: dict[str, Any] | None = None
    problem = ""
    try:
        predictor = Predictor(bundle, settings)
    except ModelUnavailableError as err:
        problem = str(err)
        log.error("Model not loaded: %s", err)
    try:
        study = load_study(bundle)
    except ValueError as err:
        log.error("Study not loaded: %s", err)

    app = FastAPI(
        title="Road Condition AI",
        version=__version__,
        description="Classify a road photo into six friction levels, with calibrated confidence.",
    )
    origins = [
        o.strip() for o in os.environ.get("ROADCOND_CORS_ORIGINS", "").split(",") if o.strip()
    ]
    if origins:  # only for a separate dev frontend; the deployed demo is same-origin
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"])

    @app.middleware("http")
    async def reject_oversized_requests(request: Request, call_next):
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > settings.max_upload_bytes + MULTIPART_OVERHEAD:
            return JSONResponse({"detail": "The upload is too large"}, status_code=413)
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    def health():
        if predictor is None:
            return JSONResponse(
                HealthResponse(
                    status="unavailable", version=__version__, detail=problem
                ).model_dump(),
                status_code=503,
            )
        return HealthResponse(version=__version__, **predictor.health())

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        file: Annotated[UploadFile, File(description="A road photo (JPEG, PNG, WebP, ...)")],
        heatmap: Annotated[bool, Query(description="Include the heatmap overlay")] = True,
    ):
        if predictor is None:
            raise HTTPException(503, "No model is loaded")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            mb = settings.max_upload_bytes / 1024 / 1024
            limit = f"{mb:g} MB" if mb >= 1 else f"{settings.max_upload_bytes // 1024} KB"
            raise HTTPException(413, f"The upload is larger than {limit}")
        try:
            return predictor.predict(data, heatmap=heatmap)
        except ImageTooLargeError as err:
            raise HTTPException(413, str(err)) from None
        except InvalidImageError as err:
            raise HTTPException(400, str(err)) from None

    @app.get("/study")
    def get_study():
        if study is None:
            raise HTTPException(404, "No study results have been added to this deployment yet")
        return study

    if (frontend / "index.html").is_file():  # last, so it can never shadow an API route
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        log.info("No built frontend at %s: serving the API only", frontend)

    return app


app = create_app()
