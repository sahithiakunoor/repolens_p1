"""
RepoLens API server — three endpoints that expose the full RAG pipeline
over HTTP, making the system demoable and integratable with any frontend.

Endpoints:
  POST /index         — ingest a public GitHub repo into the vector index
  GET  /index/{job_id} — poll an indexing job's status
  POST /query         — ask a natural-language question about an indexed repo
  GET  /health        — check the server and LLM provider are alive

Design notes:
  - Indexing is slow (clone + embed can take minutes for large repos),
    so /index runs in a background task and returns immediately with a
    job ID. Poll /index/{job_id} to check progress.
  - One Indexer and Generator per repo_url, cached in memory for the
    lifetime of the server process. Re-indexing the same URL re-uses
    the cached indexer.
  - _state_lock guards the shared _generators and _index_jobs dicts
    against concurrent modification from multiple background tasks.
  - All errors return structured JSON, not HTML — this is an API, not
    a browser app.

Run locally:
  uvicorn repolens.api:app --reload --port 8000

Or via Makefile:
  make serve
"""

import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from repolens.config import settings
from repolens.generation.generator import Generator
from repolens.ingestion.indexer import Indexer
from repolens.ingestion.loader import load_repo
from repolens.llm import llm


# ── In-memory state ───────────────────────────────────────────────────────────
# Maps repo_url → Generator (which wraps the Indexer + HybridRetriever)
_generators: dict[str, Generator] = {}

# Maps job_id → status dict for async indexing jobs
_index_jobs: dict[str, dict] = {}

# Lock to guard concurrent reads/writes to the shared dicts above.
# FastAPI's background tasks run in a thread pool — without this, two
# simultaneous /index requests could corrupt shared state.
_state_lock = threading.Lock()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    _load_persisted_indexes()
    yield


def _load_persisted_indexes():
    """
    On startup, scan the index directory for previously indexed repos and
    load them back into _generators so they're immediately queryable without
    re-indexing.

    Each subdirectory under settings.index_dir is named 'org__repo'
    (from _repo_slug). We reverse that slug back to a GitHub URL and
    instantiate a Generator pointing at the persisted ChromaDB data.
    """
    index_dir = settings.index_dir
    if not index_dir.exists():
        return

    for repo_dir in index_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        try:
            # Reverse 'org__repo' → 'https://github.com/org/repo'
            parts = repo_dir.name.split("__")
            if len(parts) != 2:
                continue
            repo_url = f"https://github.com/{parts[0]}/{parts[1]}"

            indexer = Indexer(persist_dir=str(repo_dir))
            _generators[repo_url] = Generator(indexer=indexer)
            logger.info(f"Loaded persisted index: {repo_url}")
        except Exception as e:
            logger.warning(f"Could not load index from {repo_dir.name}: {e}")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RepoLens",
    description="Ask natural-language questions about any public GitHub repository.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific origins before any public deployment
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class IndexRequest(BaseModel):
    repo_url: str
    force_reindex: bool = False   # re-index even if already cached


class IndexJobResponse(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "done" | "error"
    repo_url: str
    message: str


class IndexStatusResponse(BaseModel):
    job_id: str
    status: str
    repo_url: str
    message: str
    chunks_indexed: Optional[int] = None
    elapsed_ms: Optional[int] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    repo_url: str
    question: str


class CitationResponse(BaseModel):
    name: str
    file_path: str
    start_line: int
    end_line: int
    github_url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    intent: str
    citations: list[CitationResponse]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    indexed_repos: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Check the server and LLM provider are alive."""
    llm_status = llm.health_check()
    return HealthResponse(
        status="ok" if llm_status["ok"] else "degraded",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        indexed_repos=list(_generators.keys()),
    )


@app.post("/index", response_model=IndexJobResponse, status_code=202, tags=["indexing"])
def index_repo(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Ingest a public GitHub repo into the vector index.

    Indexing runs in the background (clone + chunk + embed can take 1-5 minutes
    for large repos). Returns a job_id immediately — poll GET /index/{job_id}
    to check progress.

    If the repo is already indexed and force_reindex=False, returns immediately
    with status "done".
    """
    repo_url = request.repo_url.rstrip("/")

    with _state_lock:
        if repo_url in _generators and not request.force_reindex:
            return IndexJobResponse(
                job_id="cached",
                status="done",
                repo_url=repo_url,
                message="Already indexed. Set force_reindex=true to re-ingest.",
            )

        job_id = str(uuid.uuid4())
        _index_jobs[job_id] = {
            "status": "pending",
            "repo_url": repo_url,
            "message": "Queued for indexing.",
            "chunks_indexed": None,
            "elapsed_ms": None,
            "error": None,
        }

    background_tasks.add_task(_run_index_job, job_id, repo_url)

    return IndexJobResponse(
        job_id=job_id,
        status="pending",
        repo_url=repo_url,
        message=f"Indexing started. Poll GET /index/{job_id} for status.",
    )


@app.get("/index/{job_id}", response_model=IndexStatusResponse, tags=["indexing"])
def index_status(job_id: str):
    """Check the status of an indexing job."""
    with _state_lock:
        if job_id not in _index_jobs:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        job = dict(_index_jobs[job_id])  # copy to release lock before returning
    return IndexStatusResponse(job_id=job_id, **job)


@app.post("/query", response_model=QueryResponse, tags=["querying"])
def query_repo(request: QueryRequest):
    """
    Ask a natural-language question about an indexed repository.

    The repo must be indexed first via POST /index. Returns a cited answer
    with GitHub line links for every chunk used to produce it.
    """
    repo_url = request.repo_url.rstrip("/")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    with _state_lock:
        generator = _generators.get(repo_url)

    if generator is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{repo_url}' is not indexed. "
                f"Call POST /index first."
            ),
        )

    response = generator.answer(request.question)

    return QueryResponse(
        answer=response.answer,
        intent=response.intent.value,
        citations=[
            CitationResponse(
                name=rc.chunk.name,
                file_path=rc.chunk.file_path,
                start_line=rc.chunk.start_line,
                end_line=rc.chunk.end_line,
                github_url=rc.chunk.github_url,
                score=round(float(rc.score), 4),
            )
            for rc in response.citations
        ],
        latency_ms=response.latency_ms,
    )


# ── Background task ───────────────────────────────────────────────────────────

def _run_index_job(job_id: str, repo_url: str):
    """Runs in the background thread pool — clones, chunks, and indexes the repo."""
    with _state_lock:
        _index_jobs[job_id]["status"] = "running"
        _index_jobs[job_id]["message"] = "Cloning and chunking repository..."
    start = time.monotonic()

    try:
        chunks, _ = load_repo(
            github_url=repo_url,
            clone_dir=str(settings.clone_dir / _repo_slug(repo_url)),
        )

        with _state_lock:
            _index_jobs[job_id]["message"] = f"Embedding {len(chunks)} chunks..."

        index_path = str(settings.index_dir / _repo_slug(repo_url))
        indexer = Indexer(persist_dir=index_path)
        indexer.index(chunks)

        generator = Generator(indexer=indexer)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        with _state_lock:
            _generators[repo_url] = generator
            _index_jobs[job_id].update({
                "status": "done",
                "message": f"Indexed {len(chunks)} chunks successfully.",
                "chunks_indexed": len(chunks),
                "elapsed_ms": elapsed_ms,
            })
        logger.info(f"Indexed {repo_url}: {len(chunks)} chunks in {elapsed_ms}ms")

    except Exception as e:
        logger.exception(f"Indexing failed for {repo_url}: {e}")
        with _state_lock:
            _index_jobs[job_id].update({
                "status": "error",
                "message": "Indexing failed.",
                "error": str(e),
            })


def _repo_slug(repo_url: str) -> str:
    """Turn 'https://github.com/org/repo' into 'org__repo' for use as a dir name."""
    parts = repo_url.rstrip("/").split("/")
    return "__".join(parts[-2:])