"""
Indexer: takes CodeChunks and builds two parallel indexes:

  1. ChromaDB (dense vector index) — for semantic search
     "how does authentication work?" finds auth-related chunks
     even if the word "authentication" doesn't appear in the code.

  2. BM25 (sparse keyword index) — for exact symbol search
     "BaseTool.run" finds that exact method name instantly.

Both indexes are queried at retrieval time and fused via RRF.

Embedding strategy:
  We embed a *contextualized* version of each chunk, not the raw source.
  Format: "{chunk_type} {name}\n{docstring}\n{content}"
  Leading with the name and docstring gives the embedding model semantic
  signal that pure code syntax lacks.
"""

import json
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from repolens.models import CodeChunk

# Embedding model — good balance of quality vs speed for code
# Swap for "voyageai/voyage-code-2" in production for better code understanding
EMBED_MODEL = "all-MiniLM-L6-v2"

# How many chunks to embed in one batch (controls memory usage)
BATCH_SIZE = 64


class Indexer:
    def __init__(self, persist_dir: str = "./repolens_index"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"🧠 Loading embedding model: {EMBED_MODEL}")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        self.chroma = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name="repolens",
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 is held in memory and pickled to disk
        self.bm25: BM25Okapi | None = None
        self.bm25_chunks: list[CodeChunk] = []  # parallel list to bm25 corpus
        self._load_bm25()

    # ── Public API ────────────────────────────────────────────────────────────

    def index(self, chunks: list[CodeChunk]) -> None:
        """Embed and store chunks in both indexes. Skips already-indexed chunks."""
        new_chunks = self._filter_new(chunks)
        if not new_chunks:
            print("ℹ️  All chunks already indexed — nothing to do.")
            return

        print(f"📥 Indexing {len(new_chunks)} new chunks...")
        self._index_chroma(new_chunks)
        self._index_bm25(new_chunks)
        print(f"✅ Index complete. Total chunks: {self.collection.count()}")

    def embed_query(self, query: str) -> list[float]:
        return self.embed_model.encode(query).tolist()

    def count(self) -> int:
        return self.collection.count()

    # ── ChromaDB ─────────────────────────────────────────────────────────────

    def _index_chroma(self, chunks: list[CodeChunk]) -> None:
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
            batch = chunks[i: i + BATCH_SIZE]
            texts      = [_contextualize(c) for c in batch]
            embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()

            self.collection.add(
                ids        =[c.chunk_id for c in batch],
                embeddings =embeddings,
                documents  =[c.content for c in batch],
                metadatas  =[_to_metadata(c) for c in batch],
            )

    def _filter_new(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Return only chunks not already in the ChromaDB collection."""
        if self.collection.count() == 0:
            return chunks
        existing_ids = set(
            self.collection.get(ids=[c.chunk_id for c in chunks])["ids"]
        )
        return [c for c in chunks if c.chunk_id not in existing_ids]

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _index_bm25(self, chunks: list[CodeChunk]) -> None:
        self.bm25_chunks.extend(chunks)
        corpus = [_bm25_text(c).split() for c in self.bm25_chunks]
        self.bm25 = BM25Okapi(corpus)
        self._save_bm25()

    def _load_bm25(self) -> None:
        bm25_path   = self.persist_dir / "bm25.pkl"
        chunks_path = self.persist_dir / "bm25_chunks.pkl"
        if bm25_path.exists() and chunks_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(chunks_path, "rb") as f:
                self.bm25_chunks = pickle.load(f)
            print(f"📂 Loaded BM25 index ({len(self.bm25_chunks)} chunks)")

    def _save_bm25(self) -> None:
        with open(self.persist_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.persist_dir / "bm25_chunks.pkl", "wb") as f:
            pickle.dump(self.bm25_chunks, f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _contextualize(chunk: CodeChunk) -> str:
    """
    Build the text that gets embedded.
    Leading with name + docstring gives semantic signal beyond raw syntax.
    """
    parts = [f"{chunk.chunk_type} {chunk.name}"]
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.content)
    return "\n".join(parts)


def _bm25_text(chunk: CodeChunk) -> str:
    """
    Text for BM25 keyword index — emphasises symbol names
    by repeating them so exact-match queries rank them highly.
    """
    return f"{chunk.name} {chunk.name} {chunk.docstring} {chunk.content}"


def _to_metadata(chunk: CodeChunk) -> dict:
    """Flatten CodeChunk to ChromaDB-compatible metadata dict (str/int/float only)."""
    return {
        "chunk_type":   chunk.chunk_type,
        "name":         chunk.name,
        "file_path":    chunk.file_path,
        "language":     chunk.language,
        "start_line":   chunk.start_line,
        "end_line":     chunk.end_line,
        "docstring":    chunk.docstring[:500] if chunk.docstring else "",
        "parent_class": chunk.parent_class,
        "repo_url":     chunk.repo_url,
        "github_url":   chunk.github_url,
        "imports":      json.dumps(chunk.imports[:10]),  # cap for metadata size
    }
