"""
Unit tests for repolens.retrieval.retriever.HybridRetriever

Everything that would require a real model (embeddings, ChromaDB, the
cross-encoder) is mocked, so these tests run fast and offline. They check
the *logic* — RRF fusion math, dedup, metadata reconstruction, reranking
order — not the underlying ML.
"""

from unittest.mock import MagicMock, patch

import pytest

from repolens.models import CodeChunk
from repolens.retrieval.retriever import HybridRetriever, _chunk_from_chroma


def make_chunk(chunk_id: str, name: str = "some_func") -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        content=f"def {name}(): pass",
        chunk_type="function",
        name=name,
        file_path="src/app.py",
        language="python",
        start_line=1,
        end_line=2,
        docstring="",
        parent_class="",
        imports=[],
        repo_url="https://github.com/org/repo",
        github_url=f"https://github.com/org/repo/blob/main/src/app.py#L1-L2",
    )


@pytest.fixture
def mock_indexer():
    indexer = MagicMock()
    indexer.embed_query.return_value = [0.1, 0.2, 0.3]
    return indexer


@pytest.fixture
def retriever(mock_indexer):
    # Patch CrossEncoder so __init__ never tries to download a real model.
    with patch("repolens.retrieval.retriever.CrossEncoder") as mock_ce_cls:
        mock_ce_cls.return_value = MagicMock()
        return HybridRetriever(indexer=mock_indexer)


# ── _chunk_from_chroma: metadata round-trip ───────────────────────────────────

def test_chunk_from_chroma_reconstructs_chunk():
    metadata = {
        "chunk_type": "method",
        "name": "Foo.bar",
        "file_path": "src/foo.py",
        "language": "python",
        "start_line": 10,
        "end_line": 20,
        "docstring": "Does a thing.",
        "parent_class": "Foo",
        "repo_url": "https://github.com/org/repo",
        "github_url": "https://github.com/org/repo/blob/main/src/foo.py#L10-L20",
        "imports": '["os", "json"]',
    }
    chunk = _chunk_from_chroma("chunk_1", metadata, "def bar(self): pass")

    assert chunk.chunk_id == "chunk_1"
    assert chunk.content == "def bar(self): pass"
    assert chunk.name == "Foo.bar"
    assert chunk.imports == ["os", "json"]  # JSON string parsed back to list
    assert chunk.start_line == 10


# ── RRF fusion ────────────────────────────────────────────────────────────────

def test_rrf_fuse_prioritizes_chunks_ranked_highly_in_both_lists(retriever):
    a, b, c = make_chunk("a"), make_chunk("b"), make_chunk("c")

    dense_ranked = [a, b, c]    # a=rank1, b=rank2, c=rank3
    sparse_ranked = [b, a, c]  # b=rank1, a=rank2, c=rank3

    fused = retriever._rrf_fuse([dense_ranked, sparse_ranked])

    # a and b each appear at rank 1 in one list and rank 2 in the other —
    # they should score higher than c, which is rank 3 in both.
    fused_ids = [chunk.chunk_id for chunk in fused]
    assert fused_ids.index("c") == 2  # c should rank last
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_fuse_dedupes_same_chunk_across_lists(retriever):
    a = make_chunk("a")
    fused = retriever._rrf_fuse([[a], [a], [a]])
    assert len(fused) == 1
    assert fused[0].chunk_id == "a"


def test_rrf_fuse_handles_empty_list(retriever):
    a = make_chunk("a")
    fused = retriever._rrf_fuse([[a], []])
    assert len(fused) == 1


def test_rrf_fuse_handles_chunk_only_in_one_list(retriever):
    a, b = make_chunk("a"), make_chunk("b")
    # 'a' appears in both lists, 'b' only in one — 'a' should outrank 'b'.
    fused = retriever._rrf_fuse([[a, b], [a]])
    fused_ids = [chunk.chunk_id for chunk in fused]
    assert fused_ids[0] == "a"


# ── Sparse search ─────────────────────────────────────────────────────────────

def test_sparse_search_returns_empty_when_bm25_not_built(mock_indexer, retriever):
    mock_indexer.bm25 = None
    mock_indexer.bm25_chunks = []
    result = retriever._sparse_search("some query", top_k=5)
    assert result == []


def test_sparse_search_filters_out_zero_score_matches(mock_indexer, retriever):
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    mock_indexer.bm25_chunks = chunks
    mock_indexer.bm25 = MagicMock()
    # 'a' and 'c' get real scores, 'b' scores 0 — should be filtered out.
    mock_indexer.bm25.get_scores.return_value = [2.5, 0.0, 1.0]

    result = retriever._sparse_search("query text", top_k=5)
    result_ids = [chunk.chunk_id for chunk in result]

    assert "b" not in result_ids
    assert result_ids[0] == "a"  # highest score first


# ── Reranking ──────────────────────────────────────────────────────────────────

def test_rerank_orders_by_cross_encoder_score_descending(retriever):
    a, b, c = make_chunk("a"), make_chunk("b"), make_chunk("c")
    # Cross-encoder gives b the highest score, despite input order a,b,c.
    retriever.reranker.predict.return_value = [0.1, 0.9, 0.5]

    result = retriever._rerank("some query", [a, b, c], top_n=3)

    assert [r.chunk.chunk_id for r in result] == ["b", "c", "a"]
    assert result[0].score == pytest.approx(0.9)


def test_rerank_respects_top_n(retriever):
    chunks = [make_chunk(str(i)) for i in range(5)]
    retriever.reranker.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

    result = retriever._rerank("query", chunks, top_n=2)
    assert len(result) == 2
    assert result[0].chunk.chunk_id == "4"  # highest score
    assert result[1].chunk.chunk_id == "3"


def test_rerank_handles_empty_candidates(retriever):
    result = retriever._rerank("query", [], top_n=5)
    assert result == []


# ── retrieve(): end-to-end with everything mocked ─────────────────────────────

def test_retrieve_combines_dense_sparse_rrf_and_rerank(mock_indexer, retriever):
    a, b = make_chunk("a"), make_chunk("b")

    # Dense search returns chroma-shaped results for chunk 'a'.
    mock_indexer.collection.query.return_value = {
        "ids": [["a"]],
        "documents": [[a.content]],
        "metadatas": [[{
            "chunk_type": a.chunk_type, "name": a.name, "file_path": a.file_path,
            "language": a.language, "start_line": a.start_line, "end_line": a.end_line,
            "docstring": a.docstring, "parent_class": a.parent_class,
            "repo_url": a.repo_url, "github_url": a.github_url,
            "imports": "[]",
        }]],
    }

    # Sparse search returns chunk 'b' directly (already a full CodeChunk).
    mock_indexer.bm25_chunks = [b]
    mock_indexer.bm25 = MagicMock()
    mock_indexer.bm25.get_scores.return_value = [3.0]

    retriever.reranker.predict.return_value = [0.7, 0.4]

    result = retriever.retrieve("some query", top_k=5, rerank_top_n=2)

    assert len(result) == 2
    returned_ids = {r.chunk.chunk_id for r in result}
    assert returned_ids == {"a", "b"}


# ── Source file bias ───────────────────────────────────────────────────────────

from repolens.retrieval.retriever import _test_file_penalty


def test_test_file_gets_penalty():
    chunk = make_chunk("a")
    chunk.file_path = "tests/test_routing.py"
    assert _test_file_penalty(chunk, 1.5) == 1.5


def test_source_file_gets_no_penalty():
    chunk = make_chunk("a")
    chunk.file_path = "src/routing.py"
    assert _test_file_penalty(chunk, 1.5) == 0.0


def test_rerank_source_file_outranks_test_file_with_equal_scores(retriever):
    source = make_chunk("source", name="real_func")
    source.file_path = "src/app.py"
    test = make_chunk("test", name="test_func")
    test.file_path = "tests/test_app.py"

    # Cross-encoder gives both the same raw score — bias should push source up
    retriever.reranker.predict.return_value = [0.5, 0.5]

    result = retriever._rerank("how does routing work?", [source, test], top_n=2)
    assert result[0].chunk.chunk_id == "source"
    assert result[1].chunk.chunk_id == "test"


def test_rerank_test_file_still_wins_if_score_much_higher(retriever):
    source = make_chunk("source", name="real_func")
    source.file_path = "src/app.py"
    test = make_chunk("test", name="test_func")
    test.file_path = "tests/test_app.py"

    # Test file score is 3.0 higher than source — penalty is only 1.5, so test wins
    retriever.reranker.predict.return_value = [0.0, 3.0]

    result = retriever._rerank("show me an example", [source, test], top_n=2)
    assert result[0].chunk.chunk_id == "test"