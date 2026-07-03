"""
Shared chunk-text helpers used by BOTH the indexer and the retriever.

These live here, in one place, because the indexer and retriever must
represent a chunk identically or retrieval quality silently degrades:

  - contextualize_chunk() builds the text the indexer EMBEDS and the text
    the cross-encoder reranker SCORES. If these two ever diverged, the
    reranker would score a different representation than what was indexed.

  - tokenize_code() builds the BM25 corpus (indexer) and tokenizes the
    query (retriever). If these used different tokenization, keyword
    matches would fail — a lowercased query would never match a
    mixed-case corpus token.

Keeping both in a single module guarantees the two sides stay in sync.
"""

import re

from repolens.models import CodeChunk


# ── Embedding / rerank representation ──────────────────────────────────────────

def contextualize_chunk(chunk: CodeChunk) -> str:
    """
    Build the canonical text representation of a chunk.

    Leading with the chunk type + name + docstring gives the embedding
    model and the cross-encoder semantic signal that raw code syntax
    alone lacks. Used by:
      - Indexer._index_chroma  (the text that gets embedded)
      - HybridRetriever._rerank (the text the cross-encoder scores)
    """
    parts = [f"{chunk.chunk_type} {chunk.name}"]
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.content)
    return "\n".join(parts)


# ── BM25 tokenization ──────────────────────────────────────────────────────────

_CAMEL_BOUNDARY_1 = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')     # getUser   -> get User
_CAMEL_BOUNDARY_2 = re.compile(r'(?<=[A-Z])(?=[A-Z][a-z])')   # APIRouter -> API Router
_NON_ALNUM = re.compile(r'[^a-zA-Z0-9]+')


def tokenize_code(text: str) -> list[str]:
    """
    Tokenize code or a natural-language query into normalized keyword tokens.

    Splits camelCase and PascalCase, then splits on every non-alphanumeric
    character (handling snake_case, dots, parens, colons), lowercases, and
    drops single-character tokens.

    Must be used by BOTH the indexer (corpus) and retriever (query), or
    keyword matching breaks on case and punctuation.
    """
    text = _CAMEL_BOUNDARY_1.sub(' ', text)
    text = _CAMEL_BOUNDARY_2.sub(' ', text)
    tokens = _NON_ALNUM.split(text)
    return [t.lower() for t in tokens if len(t) > 1]