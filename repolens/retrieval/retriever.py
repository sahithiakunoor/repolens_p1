"""
Hybrid retrieval: combines dense (ChromaDB) and sparse (BM25) search results
via Reciprocal Rank Fusion, then reranks the fused candidates with a
cross-encoder for the final ordering.

Why hybrid instead of just one method:
  - Dense embeddings are great at semantic match ("how does auth work" finds
    auth-related code even without the word "auth") but can miss exact
    symbol names buried in a sea of semantically-similar chunks.
  - BM25 is great at exact keyword/symbol match ("BaseTool.run") but has no
    notion of meaning — it can't connect a query to code that uses different
    words for the same concept.
  - Neither alone is reliably best across all query types, so we run both
    and fuse.

Why RRF (Reciprocal Rank Fusion) for combining them:
  Dense similarity scores (cosine distance) and BM25 scores live on
  completely different scales, so you can't just add them together
  meaningfully. RRF sidesteps this by ignoring raw scores entirely and
  fusing based on *rank position* in each list:
      rrf_score(doc) = sum over each ranking list containing doc of:
                          1 / (k + rank_in_that_list)
  A document ranked highly in either (or both) lists gets a high fused
  score, with no scale-matching required. k (default 60, from the
  original RRF paper) dampens the impact of any single rank-1 hit so
  lower ranks still contribute meaningfully.

Why a cross-encoder rerank on top of RRF:
  RRF fusion is still based on cheap first-pass scores (embedding cosine
  similarity, BM25 term overlap). A cross-encoder scores the query and a
  candidate chunk *together* in one forward pass, letting it model
  query-chunk interaction directly rather than comparing two separately-
  computed representations. That's significantly more accurate, but too
  slow to run over an entire corpus — so it only reranks the small
  RRF-fused candidate set (retrieval_top_k), not everything.
"""

import json

from loguru import logger
from sentence_transformers import CrossEncoder

from repolens.config import settings
from repolens.ingestion.indexer import Indexer
from repolens.models import CodeChunk, RetrievedChunk
from repolens.chunk_text import contextualize_chunk, tokenize_code


# Paths that indicate test/fixture/doc files — ranked below source files
# when cross-encoder scores are close. Configurable via settings.test_file_penalty.
_NON_SOURCE_PATTERNS = (
    "/tests/",
    "/test_",
    "test_",
    "/docs/",
    "/examples/",
    "/fixtures/",
    "/benchmarks/",
)


class HybridRetriever:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        logger.info(f"Loading reranker: {settings.rerank_model}")
        self.reranker = CrossEncoder(settings.rerank_model)

    # ── Public API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = settings.retrieval_top_k,
        rerank_top_n: int = settings.rerank_top_n,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval for a query:
          1. Dense search (ChromaDB) + sparse search (BM25), each top_k.
          2. Fuse the two ranked lists via RRF.
          3. Cross-encoder rerank the fused candidates.
          4. Return the top rerank_top_n as RetrievedChunk, scored and ordered.
        """
        dense_chunks = self._dense_search(query, top_k)
        sparse_chunks = self._sparse_search(query, top_k)

        fused = self._rrf_fuse([dense_chunks, sparse_chunks])
        candidates = fused[:top_k]

        return self._rerank(query, candidates, rerank_top_n)

    # ── Dense search (ChromaDB) ──────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int) -> list[CodeChunk]:
        embedding = self.indexer.embed_query(query)
        results = self.indexer.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        return [
            _chunk_from_chroma(chunk_id, metadata, document)
            for chunk_id, metadata, document in zip(ids, metadatas, documents)
        ]

    # ── Sparse search (BM25) ─────────────────────────────────────────────────

    def _sparse_search(self, query: str, top_k: int) -> list[CodeChunk]:
        if self.indexer.bm25 is None or not self.indexer.bm25_chunks:
            return []
        tokenized_query = tokenize_code(query)
        scores = self.indexer.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.indexer.bm25_chunks[i] for i in ranked_indices if scores[i] > 0]

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _rrf_fuse(self, ranked_lists: list[list[CodeChunk]]) -> list[CodeChunk]:
        """
        Fuse multiple ranked lists of CodeChunks by Reciprocal Rank Fusion.
        Returns a single list of CodeChunks ordered by fused score, descending.
        """
        k = settings.rrf_k
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, CodeChunk] = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
                chunk_by_id.setdefault(chunk.chunk_id, chunk)

        ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [chunk_by_id[cid] for cid in ordered_ids]

    # ── Cross-encoder reranking ───────────────────────────────────────────────

    def _rerank(
        self, query: str, candidates: list[CodeChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, contextualize_chunk(chunk)) for chunk in candidates]
        scores = self.reranker.predict(pairs)

        # Apply source file bias: penalize test/fixture/doc files so that
        # source code ranks above test code when scores are close.
        # The penalty comes from settings so it's tunable without code changes.
        penalty = settings.test_file_penalty
        adjusted_scores = [
            float(score) - (_test_file_penalty(chunk, penalty))
            for chunk, score in zip(candidates, scores)
        ]

        scored = list(zip(candidates, adjusted_scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [
            RetrievedChunk(chunk=chunk, score=round(score, 4))
            for chunk, score in scored[:top_n]
        ]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chunk_from_chroma(chunk_id: str, metadata: dict, document: str) -> CodeChunk:
    """Reconstruct a CodeChunk from a ChromaDB query result row."""
    return CodeChunk(
        chunk_id=chunk_id,
        content=document,
        chunk_type=metadata["chunk_type"],
        name=metadata["name"],
        file_path=metadata["file_path"],
        language=metadata["language"],
        start_line=metadata["start_line"],
        end_line=metadata["end_line"],
        docstring=metadata["docstring"],
        parent_class=metadata["parent_class"],
        imports=json.loads(metadata["imports"]),
        repo_url=metadata["repo_url"],
        github_url=metadata["github_url"],
    )




def _test_file_penalty(chunk: CodeChunk, penalty: float) -> float:
    """
    Return a score penalty for chunks from test/doc/example files.
    Source files get 0 penalty. Test/fixture/doc files get the configured penalty.

    This is a soft bias, not a hard filter — a test file that's genuinely
    the best match for a query (e.g. "show me an example of X") can still
    outrank a weakly relevant source file if its cross-encoder score is
    more than `penalty` points higher.
    """
    path = chunk.file_path.lower()
    if any(pattern in path for pattern in _NON_SOURCE_PATTERNS):
        return penalty
    return 0.0