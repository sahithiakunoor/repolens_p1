"""
Central configuration — all settings come from environment variables,
with sensible defaults so the app works out of the box without any .env file.

Default setup is 100% free:
  - LLM:        Ollama (local) with llama3.2:3b
  - Embeddings: sentence-transformers all-MiniLM-L6-v2
  - Vector DB:  ChromaDB (local, no server needed)

To switch to paid providers later, just update .env:
  LLM_PROVIDER=openai  +  OPENAI_API_KEY=sk-...
  LLM_PROVIDER=groq    +  GROQ_API_KEY=gsk_...

Usage anywhere in the codebase:
    from repolens.config import settings
    model = settings.llm_model
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── LLM provider ──────────────────────────────────────────────────────────
    # "mistral" (free, code-trained) ← DEFAULT
    # "gemini"  (free, 1M context)
    # "groq"    (free, fastest)
    # "ollama"  (free, local)
    # "openai"  (paid)
    llm_provider: str = os.getenv("LLM_PROVIDER", "mistral")

    # Model name — interpreted by the active provider:
    # mistral: "codestral-latest", "mistral-small-latest", "open-mistral-nemo"
    # gemini:  "gemini-2.5-flash", "gemini-2.0-flash"
    # groq:    "llama-3.1-8b-instant", "llama-3.3-70b-versatile"
    # ollama:  "llama3.2:3b", "llama3.1:8b", "mistral"
    # openai:  "gpt-4o-mini", "gpt-4o"
    llm_model: str = os.getenv("LLM_MODEL", "codestral-latest")

    # ── API keys (only set the one you need) ──────────────────────────────────
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    gemini_api_key:  str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key:    str = os.getenv("GROQ_API_KEY", "")
    openai_api_key:  str = os.getenv("OPENAI_API_KEY", "")

    # Ollama base URL — default works locally and on Colab with tunnel
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Free, runs locally via sentence-transformers
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    # How many chunks to embed in one batch (controls memory vs. speed tradeoff)
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", 64))

    # ── Reranker ──────────────────────────────────────────────────────────────
    # Free cross-encoder, runs locally
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

    # Score penalty applied to test/doc/fixture files during reranking.
    # Soft bias — a genuinely better test file can still win if its
    # cross-encoder score exceeds a source file by more than this margin.
    test_file_penalty: float = float(os.getenv("TEST_FILE_PENALTY", 1.5))

    # ── Paths ─────────────────────────────────────────────────────────────────
    index_dir: Path = Path(os.getenv("INDEX_DIR", ".repolens_index"))
    clone_dir: Path = Path(os.getenv("CLONE_DIR", ".repolens_repos"))

    # ── Ingestion ─────────────────────────────────────────────────────────────
    max_file_bytes:      int = int(os.getenv("MAX_FILE_BYTES", 200_000))
    # Metadata size caps — ChromaDB stores metadata as strings; keep them bounded
    max_imports:         int = int(os.getenv("MAX_IMPORTS", 10))
    max_docstring_chars: int = int(os.getenv("MAX_DOCSTRING_CHARS", 500))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", 20))
    rerank_top_n:    int = int(os.getenv("RERANK_TOP_N", 5))
    rrf_k:           int = int(os.getenv("RRF_K", 60))

    # ── Server ────────────────────────────────────────────────────────────────
    port:      int = int(os.getenv("PORT", 8000))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        """Raise early with a clear message if required config is missing."""
        required_keys = {
            "mistral": ("MISTRAL_API_KEY", self.mistral_api_key, "https://console.mistral.ai"),
            "gemini":  ("GEMINI_API_KEY",  self.gemini_api_key,  "https://aistudio.google.com"),
            "groq":    ("GROQ_API_KEY",    self.groq_api_key,    "https://console.groq.com"),
            "openai":  ("OPENAI_API_KEY",  self.openai_api_key,  "https://platform.openai.com"),
        }
        if self.llm_provider in required_keys:
            env_var, value, signup_url = required_keys[self.llm_provider]
            if not value:
                raise EnvironmentError(
                    f"LLM_PROVIDER={self.llm_provider} but {env_var} is not set.\n"
                    f"Add it to your .env file:\n"
                    f"  {env_var}=your_key_here\n"
                    f"Get a free key at: {signup_url}"
                )


settings = Settings()