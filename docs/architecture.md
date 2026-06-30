# RepoLens — Architecture

## Design principles

1. **AST-first, not token-first** — code structure determines chunk boundaries, not token count
2. **Intent-aware end to end** — the classified intent affects retrieval filters, reranking, AND the prompt template
3. **Fail gracefully** — unsupported languages fall back to whole-file chunking rather than crashing
4. **Config over code** — every tunable parameter lives in `.env`, not hardcoded

## Data flow

```
GitHub URL
  └─ loader.py         shallow git clone, file walk, size filter
      └─ chunker.py    tree-sitter AST → CodeChunk list
          └─ indexer.py
              ├─ ChromaDB   voyage-code-2 / MiniLM embeddings
              └─ BM25       symbol name + docstring keyword index

User query
  └─ intent.py         zero-shot LLM classification → QueryIntent
      └─ retriever.py
          ├─ ChromaDB query  (dense, top-20, with metadata filter)
          ├─ BM25 query      (sparse, top-20)
          └─ RRF fusion      → merged top-20
              └─ reranker.py  cross-encoder → top-5
                  └─ generator.py
                      ├─ prompts.py   intent-aware template
                      └─ LLM call     → answer + GitHub citations
```

## Why hybrid retrieval?

Dense search alone misses exact symbol names ("BaseTool" → finds similar classes but not the exact one).
BM25 alone misses semantic queries ("how does retry work?" → no file called "retry").
Hybrid + RRF consistently outperforms either alone by 5–15% recall.

## Why AST chunking?

Token-based chunking: a 200-line function gets split at line 128 — the second chunk has no function signature, no docstring, and is essentially meaningless to retrieve.
AST chunking: the function is one chunk, regardless of length, with name and docstring as first-class metadata.

## Intent → retrieval strategy mapping

| Intent | Metadata filter | Retrieval bias | Prompt style |
|--------|----------------|----------------|-------------|
| EXPLAIN | none | semantic dense | step-by-step explanation |
| EXAMPLE | chunk_type=function | dense + imports in context | runnable snippet |
| FIND | chunk_type=function/class | BM25 upweighted | file path + line |
| DEBUG | none | dense, broader top-k | diagnostic analysis |
| COMPARE | none | dense, fetch both entities | side-by-side comparison |
