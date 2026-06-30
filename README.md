# RepoLens 🔍

> **Query any GitHub repository in natural language.**
> Ask "how does authentication work?" or "show me an example of retry logic" and get cited, grounded answers with links to exact lines of code.

[![CI](https://github.com/yourusername/repolens/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/repolens/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What makes RepoLens different

Most code search tools (GitHub search, grep, ctags) match keywords. RepoLens **understands meaning**:

| Query | GitHub search | RepoLens |
|-------|--------------|---------|
| "how does retry logic work?" | No results (no file called "retry logic") | Finds `RetryHandler.execute()`, explains the backoff strategy |
| "show me an example of streaming" | Files containing "streaming" | Returns a runnable code snippet with citations |
| "where is BaseChain defined?" | Lists all files | Returns exact file path + line number + GitHub link |

**Key differentiators vs. other RAG-over-code projects:**

- ✅ **AST/tree-sitter chunking** — functions and classes as atomic units, not arbitrary token windows
- ✅ **Intent classification** — 5 intents (explain / example / find / debug / compare), each with its own retrieval strategy and prompt
- ✅ **Hybrid retrieval** — dense semantic search + BM25 keyword search fused via Reciprocal Rank Fusion
- ✅ **Cross-encoder reranking** — precision top-5 from a 20-candidate pool
- ✅ **Clickable GitHub citations** — every answer includes exact line links to the real repo
- ✅ **RAGAS evaluation harness** — quantified metrics, not vibes

---

## Architecture

```
GitHub URL
    │
    ▼
┌─────────────────────────────────┐
│  INGESTION                      │
│  clone → walk → AST chunk       │
│  → embed → ChromaDB + BM25      │
└───────────────┬─────────────────┘
                │
    User query  │
        │       ▼
┌───────────────────────────────────┐
│  RETRIEVAL                        │
│  classify intent                  │
│  → dense search + BM25            │
│  → RRF fusion                     │
│  → cross-encoder rerank           │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  GENERATION                       │
│  intent-aware prompt              │
│  → GPT-4o-mini                    │
│  → answer + GitHub line citations │
└───────────────────────────────────┘
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/repolens
cd repolens
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest a repository

```bash
# Via make
make ingest REPO=https://github.com/tiangolo/fastapi

# Or directly
python -m repolens.scripts.ingest --repo https://github.com/tiangolo/fastapi
```

### 4. Launch the UI

```bash
make run-ui
# Opens http://localhost:8501
```

---

## Supported languages

| Language | Extension | Chunking |
|----------|-----------|---------|
| Python | `.py` | `ast` (stdlib) |
| JavaScript | `.js` | tree-sitter |
| TypeScript | `.ts`, `.tsx` | tree-sitter |
| Java | `.java` | tree-sitter |
| Go | `.go` | tree-sitter |

---

## Project structure

```
repolens/
├── repolens/                  # Main package
│   ├── config.py              # All settings via env vars
│   ├── models.py              # Shared dataclasses
│   ├── ingestion/
│   │   ├── chunker.py         # AST/tree-sitter chunker (core)
│   │   ├── loader.py          # GitHub cloner + file walker
│   │   └── indexer.py         # ChromaDB + BM25 indexer
│   ├── retrieval/
│   │   ├── intent.py          # Query intent classifier
│   │   ├── retriever.py       # Hybrid dense + BM25 + RRF
│   │   └── reranker.py        # Cross-encoder reranker
│   ├── generation/
│   │   ├── prompts.py         # Intent-aware prompt templates
│   │   └── generator.py       # LLM reader + citation builder
│   ├── evaluation/
│   │   ├── golden_set.py      # 30-question ground truth
│   │   └── evaluator.py       # RAGAS + custom metrics
│   └── scripts/
│       ├── ingest.py          # CLI: ingest a repo
│       └── serve.py           # CLI: start API server
├── tests/
│   ├── unit/                  # Fast, no network, no LLM
│   └── integration/           # End-to-end with real repos
├── docs/
│   ├── architecture.md
│   └── evaluation_results.md
├── config/                    # Static config files
├── app.py                     # Streamlit entry point
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Evaluation

RepoLens ships with a 30-question golden test set covering all 5 intents. Run evaluation with:

```bash
pytest tests/integration/test_evaluation.py -v
```

| Metric | Target | Achieved |
|--------|--------|---------|
| Faithfulness (RAGAS) | ≥ 0.85 | TBD |
| Answer relevance | ≥ 0.80 | TBD |
| Context recall | ≥ 0.75 | TBD |
| MRR@10 (retrieval) | ≥ 0.70 | TBD |
| P95 latency | < 3s | TBD |

---

## Deployment

### Railway (recommended)

```bash
railway login
railway new
railway up
```

Set `OPENAI_API_KEY` in Railway's environment variables dashboard.

### Docker

```bash
make docker-build
docker-compose up
```

---

## License

MIT — see [LICENSE](LICENSE).
