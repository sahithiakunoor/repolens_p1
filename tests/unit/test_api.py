"""
Unit tests for repolens.api

Uses FastAPI's TestClient so we can make real HTTP calls against the app
without actually starting a server. All external calls (indexing, LLM,
retrieval) are mocked — we're testing routing, status codes, request
validation, and response shapes, not the underlying pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from repolens.api import app, _generators, _index_jobs
from repolens.models import QueryIntent, RAGResponse, RetrievedChunk, CodeChunk


@pytest.fixture(autouse=True)
def clear_state():
    """Reset in-memory state between tests so they don't bleed into each other."""
    _generators.clear()
    _index_jobs.clear()
    yield
    _generators.clear()
    _index_jobs.clear()


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def make_retrieved_chunk(name="my_func", file_path="src/app.py") -> RetrievedChunk:
    chunk = CodeChunk(
        chunk_id="chunk_1",
        content="def my_func(): pass",
        chunk_type="function",
        name=name,
        file_path=file_path,
        language="python",
        start_line=10,
        end_line=20,
        docstring="",
        parent_class="",
        imports=[],
        repo_url="https://github.com/org/repo",
        github_url="https://github.com/org/repo/blob/main/src/app.py#L10-L20",
    )
    return RetrievedChunk(chunk=chunk, score=0.95)


# ── GET /health ────────────────────────────────────────────────────────────────

def test_health_returns_ok_when_llm_healthy(client):
    with patch("repolens.api.llm") as mock_llm:
        mock_llm.health_check.return_value = {"ok": True}
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_provider" in data
    assert "indexed_repos" in data


def test_health_returns_degraded_when_llm_down(client):
    with patch("repolens.api.llm") as mock_llm:
        mock_llm.health_check.return_value = {"ok": False}
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_health_lists_indexed_repos(client):
    _generators["https://github.com/org/repo"] = MagicMock()
    with patch("repolens.api.llm") as mock_llm:
        mock_llm.health_check.return_value = {"ok": True}
        response = client.get("/health")
    assert "https://github.com/org/repo" in response.json()["indexed_repos"]


# ── POST /index ───────────────────────────────────────────────────────────────

def test_index_returns_202_and_job_id(client):
    with patch("repolens.api._run_index_job"):
        response = client.post("/index", json={"repo_url": "https://github.com/org/repo"})
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_index_already_cached_returns_done(client):
    _generators["https://github.com/org/repo"] = MagicMock()
    response = client.post("/index", json={"repo_url": "https://github.com/org/repo"})
    assert response.status_code == 202
    assert response.json()["status"] == "done"
    assert response.json()["job_id"] == "cached"


def test_index_force_reindex_bypasses_cache(client):
    _generators["https://github.com/org/repo"] = MagicMock()
    with patch("repolens.api._run_index_job"):
        response = client.post(
            "/index",
            json={"repo_url": "https://github.com/org/repo", "force_reindex": True},
        )
    assert response.json()["status"] == "pending"


# ── GET /index/{job_id} ───────────────────────────────────────────────────────

def test_index_status_returns_job_state(client):
    _index_jobs["abc123"] = {
        "status": "running",
        "repo_url": "https://github.com/org/repo",
        "message": "Embedding chunks...",
        "chunks_indexed": None,
        "elapsed_ms": None,
        "error": None,
    }
    response = client.get("/index/abc123")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_index_status_404_for_unknown_job(client):
    response = client.get("/index/nonexistent")
    assert response.status_code == 404


# ── POST /query ───────────────────────────────────────────────────────────────

def test_query_returns_answer_and_citations(client):
    mock_gen = MagicMock()
    mock_gen.answer.return_value = RAGResponse(
        answer="The function does X.",
        citations=[make_retrieved_chunk()],
        intent=QueryIntent.EXPLAIN,
        latency_ms=120,
    )
    _generators["https://github.com/org/repo"] = mock_gen

    response = client.post(
        "/query",
        json={
            "repo_url": "https://github.com/org/repo",
            "question": "how does my_func work?",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The function does X."
    assert data["intent"] == "explain"
    assert len(data["citations"]) == 1
    assert "github_url" in data["citations"][0]
    assert data["latency_ms"] == 120


def test_query_404_when_repo_not_indexed(client):
    response = client.post(
        "/query",
        json={
            "repo_url": "https://github.com/org/not-indexed",
            "question": "how does this work?",
        },
    )
    assert response.status_code == 404
    assert "not indexed" in response.json()["detail"].lower()


def test_query_400_for_empty_question(client):
    _generators["https://github.com/org/repo"] = MagicMock()
    response = client.post(
        "/query",
        json={"repo_url": "https://github.com/org/repo", "question": "   "},
    )
    assert response.status_code == 400


def test_query_citation_fields_are_complete(client):
    mock_gen = MagicMock()
    mock_gen.answer.return_value = RAGResponse(
        answer="Answer.",
        citations=[make_retrieved_chunk(name="some_func", file_path="src/utils.py")],
        intent=QueryIntent.FIND,
        latency_ms=50,
    )
    _generators["https://github.com/org/repo"] = mock_gen

    response = client.post(
        "/query",
        json={"repo_url": "https://github.com/org/repo", "question": "where is it?"},
    )
    citation = response.json()["citations"][0]
    assert citation["name"] == "some_func"
    assert citation["file_path"] == "src/utils.py"
    assert citation["start_line"] == 10
    assert citation["end_line"] == 20
    assert citation["score"] == 0.95