# Road Condition AI demo: React frontend + FastAPI + ONNX Runtime (CPU) in one image.
# Hugging Face Spaces (Docker SDK) compatible: port 7860, non-root user with uid 1000.
#
#   docker build -t roadcond .
#   docker run -p 7860:7860 -v "$PWD/serving:/app/serving:ro" roadcond
#
# The model bundle (serving/model.onnx, meta.json, study.json) is not in git. Either mount it
# as above or put it in serving/ before building. Without it the page loads and says no model.

# --- stage 1: build the frontend ------------------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- stage 2: the service (no torch: ONNX Runtime, Pillow, FastAPI only) --------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --root-user-action=ignore ".[serve]"

COPY --chown=user api ./api
COPY --chown=user serving ./serving
COPY --from=frontend --chown=user /build/dist ./frontend/dist

USER user
ENV ROADCOND_BUNDLE_DIR=/app/serving \
    ROADCOND_FRONTEND_DIR=/app/frontend/dist
EXPOSE 7860

# Liveness, not readiness: /health answers 503 when no model is loaded, which is not a crash.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/openapi.json', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
