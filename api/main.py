"""Demo API (PRD W1, W2): road photo in, calibrated condition, uncertainty and heatmap out.

    ROADCOND_BUNDLE_DIR=serving uvicorn api.main:app

The model comes from a bundle folder written by `python -m roadcond.export`; research and
serving meet only there. If the bundle is missing the app still starts, /health and /predict
answer 503, and /study keeps working when a study.json is present.

When a built frontend exists (frontend/dist, or $ROADCOND_FRONTEND_DIR) it is served at "/"
after every API route, so the demo is one same-origin service.

Review routing (roadcond.routing.live) is on when the bundle has a routing.json checked on its
own model: every /predict answer then carries a review decision, appended to the decision log
($ROADCOND_DECISION_LOG). Reviewers named in $ROADCOND_REVIEWER_TOKENS ("name:token,...") read
the queue and record labels under /review; with $ROADCOND_REVIEW_IMAGE_DIR set, uploaded photos
are kept there so a second reviewer can see them.
"""

import hashlib
import hmac
import logging
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from roadcond import __version__
from roadcond.routing.live import (
    LiveRouting,
    LiveRoutingError,
    ReviewConflictError,
    UnknownDecisionError,
)
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
DEFAULT_DECISION_LOG = "logs/decisions.jsonl"
MIN_TOKEN_LENGTH = 16


def parse_reviewer_tokens(text: str) -> dict[str, str]:
    """ "name:token,name:token" -> {token: name}. Short or malformed entries are skipped."""
    tokens: dict[str, str] = {}
    for entry in filter(None, (e.strip() for e in text.split(","))):
        name, _, token = entry.partition(":")
        if not name.strip() or len(token) < MIN_TOKEN_LENGTH:
            log.error(
                "Skipped a reviewer token entry: need name:token with %d+ characters",
                MIN_TOKEN_LENGTH,
            )
            continue
        tokens[token] = name.strip()
    return tokens


class ModelInfo(BaseModel):
    name: str
    mock: bool


class ReviewInfo(BaseModel):
    decision_id: str | None  # null only when the decision could not be logged
    mode: str  # light_review | full_review | second_review
    reasons: list[str]
    fallback: bool
    reviews_needed: int
    recorded: bool


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
    review: ReviewInfo | None = None  # null when review routing is off


class ReviewRequest(BaseModel):
    label: str


class ReviewState(BaseModel):
    decision_id: str
    routed_at: str
    prediction: str | None
    mode: str
    reasons: list[str]
    status: str  # pending | disputed | done
    agreed_label: str | None
    reviews_needed: int
    reviewers: list[str]
    has_image: bool = False


class ReviewQueue(BaseModel):
    reviewer: str
    counts: dict[str, dict[str, int]]
    items: list[ReviewState]


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
    routing: dict[str, Any] | None = None


def create_app(
    bundle_dir: str | Path | None = None,
    settings: ServingSettings | None = None,
    frontend_dir: str | Path | None = None,
    decision_log: str | Path | None = None,
    reviewer_tokens: Mapping[str, str] | None = None,
    review_image_dir: str | Path | None = None,
) -> FastAPI:
    """Build the app around a bundle folder (default: $ROADCOND_BUNDLE_DIR or ./serving).

    frontend_dir: a built frontend to serve at "/" (default: $ROADCOND_FRONTEND_DIR, else
    frontend/dist when it has been built).
    decision_log: the routing audit log (default: $ROADCOND_DECISION_LOG, else
    logs/decisions.jsonl). reviewer_tokens: {token: reviewer name} (default: parsed from
    $ROADCOND_REVIEWER_TOKENS). review_image_dir: keep uploads for reviewers (default:
    $ROADCOND_REVIEW_IMAGE_DIR; unset keeps none).
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

    routing: LiveRouting | None = None
    routing_problem = problem or "No model is loaded"
    if predictor is not None:
        log_path = Path(
            decision_log or os.environ.get("ROADCOND_DECISION_LOG", DEFAULT_DECISION_LOG)
        )
        try:
            routing = LiveRouting(bundle, predictor.meta, log_path)
            log.info("Review routing on; decisions go to %s", log_path)
        except LiveRoutingError as err:
            routing_problem = str(err)
            log.warning("Review routing off: %s", err)
    tokens = dict(
        reviewer_tokens
        if reviewer_tokens is not None
        else parse_reviewer_tokens(os.environ.get("ROADCOND_REVIEWER_TOKENS", ""))
    )
    image_dir_value = review_image_dir or os.environ.get("ROADCOND_REVIEW_IMAGE_DIR")
    image_dir = Path(image_dir_value) if image_dir_value else None
    if image_dir is not None and routing is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    def routing_health() -> dict[str, Any]:
        if routing is None:
            return {"enabled": False, "detail": routing_problem}
        return {
            "enabled": True,
            "reviewing": bool(tokens),
            "stores_images": image_dir is not None,
            **routing.info,
        }

    def image_path(decision_id: str) -> Path | None:
        if image_dir is None:
            return None
        try:
            return image_dir / str(uuid.UUID(decision_id))  # never a path from the request
        except ValueError:
            return None

    def reviewer(authorization: Annotated[str | None, Header()] = None) -> str:
        if routing is None:
            raise HTTPException(503, "Review routing is not enabled on this deployment")
        if not tokens:
            raise HTTPException(503, "Reviewing is not enabled: no reviewer tokens are set")
        scheme, _, token = (authorization or "").partition(" ")
        name = None
        if scheme.lower() == "bearer" and token:
            for known, who in tokens.items():
                if hmac.compare_digest(known.encode(), token.encode()):
                    name = who
        if name is None:
            raise HTTPException(
                401, "A valid reviewer token is needed", headers={"WWW-Authenticate": "Bearer"}
            )
        return name

    def with_image(state: dict[str, Any]) -> dict[str, Any]:
        path = image_path(state["decision_id"])
        return {**state, "has_image": bool(path and path.is_file())}

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
        return HealthResponse(version=__version__, routing=routing_health(), **predictor.health())

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
            result = predictor.predict(data, heatmap=heatmap)
        except ImageTooLargeError as err:
            raise HTTPException(413, str(err)) from None
        except InvalidImageError as err:
            raise HTTPException(400, str(err)) from None
        result["review"] = None
        if routing is not None:
            review = routing.route(hashlib.sha256(data).hexdigest(), result["probabilities"])
            path = image_path(review["decision_id"]) if review["decision_id"] else None
            if path is not None:
                try:
                    path.write_bytes(data)
                except OSError as err:  # the decision stands; only the reviewer's copy is lost
                    log.error("Could not keep the photo for review: %s", err)
            result["review"] = review
        return result

    @app.get("/review/queue", response_model=ReviewQueue)
    def review_queue(
        name: Annotated[str, Depends(reviewer)],
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ):
        """Open decisions this reviewer has not reviewed yet, strictest mode first."""
        assert routing is not None  # reviewer() refuses otherwise
        items = [with_image(s) for s in routing.queue(name, limit)]
        return {"reviewer": name, "counts": routing.counts(), "items": items}

    @app.get("/review/{decision_id}/image")
    def review_image(decision_id: str, _: Annotated[str, Depends(reviewer)]):
        path = image_path(decision_id)
        if path is None or not path.is_file():
            raise HTTPException(404, "No photo was kept for this decision")
        try:
            with Image.open(path) as image:
                media_type = Image.MIME.get(image.format or "", "application/octet-stream")
        except OSError:
            raise HTTPException(404, "No photo was kept for this decision") from None
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post("/review/{decision_id}", response_model=ReviewState)
    def record_review(
        decision_id: str,
        body: Annotated[ReviewRequest, Body()],
        name: Annotated[str, Depends(reviewer)],
    ):
        """Record this reviewer's label. A second review is done when two reviewers agree."""
        assert routing is not None
        try:
            return with_image(routing.review(decision_id, body.label, name))
        except UnknownDecisionError:
            raise HTTPException(404, "No such decision") from None
        except ReviewConflictError as err:
            raise HTTPException(409, str(err)) from None
        except ValueError as err:
            raise HTTPException(422, str(err)) from None

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
