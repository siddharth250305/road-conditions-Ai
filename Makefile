# One command each: `make setup`, then `make demo-mock` (no training needed) or `make demo`.
PY ?= python3
PORT ?= 8000

.PHONY: setup test lint frontend demo demo-mock docker

setup:  ## Python and frontend dependencies
	$(PY) -m pip install -e ".[dev,export,serve]"
	cd frontend && npm ci

test: lint  ## Everything CI runs: ruff, pytest, frontend type check, frontend tests
	$(PY) -m pytest -q
	cd frontend && npm run typecheck && npm test

lint:
	ruff check src tests scripts api
	ruff format --check src tests scripts api

frontend:  ## Build the React page into frontend/dist (served by the API)
	cd frontend && npm run build

demo-mock: frontend  ## Demo on a random-weight model, clearly marked as mock
	$(PY) scripts/make_mock_bundle.py --out serving
	ROADCOND_BUNDLE_DIR=serving $(PY) -m uvicorn api.main:app --port $(PORT)

demo: frontend  ## Demo on the model bundle in ./serving (see scripts/export_model.py)
	ROADCOND_BUNDLE_DIR=serving $(PY) -m uvicorn api.main:app --port $(PORT)

docker:  ## The same demo as the Hugging Face Space image, on http://localhost:7860
	docker compose up --build
