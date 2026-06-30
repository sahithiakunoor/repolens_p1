.PHONY: install install-dev lint format test test-unit test-integration run-ui run-api ingest clean help

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check repolens/ tests/

format:
	ruff format repolens/ tests/

typecheck:
	mypy repolens/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-cov:
	pytest tests/ --cov=repolens --cov-report=html --cov-report=term-missing

# ── Run ───────────────────────────────────────────────────────────────────────

run-ui:
	streamlit run app.py

run-api:
	uvicorn repolens.api:app --reload --port 8000

# ── Ingestion ─────────────────────────────────────────────────────────────────

ingest:
	@echo "Usage: make ingest REPO=https://github.com/owner/repo"
	python -m repolens.scripts.ingest --repo $(REPO)

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t repolens:latest .

docker-run:
	docker-compose up

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov

clean-index:
	rm -rf .repolens_index .repolens_repos
	@echo "Index and cloned repos cleared."

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  RepoLens — available commands"
	@echo ""
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install dev dependencies + pre-commit"
	@echo "  make lint             Lint with ruff"
	@echo "  make format           Format with ruff"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make run-ui           Launch Streamlit UI"
	@echo "  make run-api          Launch FastAPI server"
	@echo "  make ingest REPO=...  Ingest a GitHub repo"
	@echo "  make docker-build     Build Docker image"
	@echo "  make clean            Remove cache files"
	@echo "  make clean-index      Remove vector index + cloned repos"
	@echo ""
