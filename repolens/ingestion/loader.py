"""
Repo loader: clones any public GitHub repo and walks its files,
feeding each source file through the chunker.

Design decisions:
  - Shallow clone (depth=1) so we don't pull git history — fast and cheap
  - Skip vendor dirs, test files, and auto-generated code by default
  - Progress bar via tqdm so the user knows it's working
  - Returns a flat list of CodeChunks ready for embedding
"""

import os
import shutil
import tempfile
from pathlib import Path

from git import Repo
from tqdm import tqdm

from repolens.models import CodeChunk, SUPPORTED_LANGUAGES
from repolens.ingestion.chunker import chunk_file, detect_language

# Directories to always skip — noise, not signal
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".eggs", "*.egg-info",
    "vendor", "third_party", "fixtures", "migrations",
}

# Max file size to parse (skip huge auto-generated files)
MAX_FILE_BYTES = 200_000  # 200 KB


def load_repo(github_url: str, clone_dir: str = None) -> tuple[list[CodeChunk], str]:
    """
    Clone a GitHub repo and extract all CodeChunks.

    Args:
        github_url: e.g. "https://github.com/tiangolo/fastapi"
        clone_dir:  optional path to clone into (temp dir used if None)

    Returns:
        (chunks, clone_path) — chunks ready for embedding, path to cloned repo
    """
    # Normalise URL — strip trailing slash and .git suffix
    url = github_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    repo_name = url.split("/")[-1]
    owner     = url.split("/")[-2]
    base_url  = f"https://github.com/{owner}/{repo_name}"

    if clone_dir is None:
        clone_dir = tempfile.mkdtemp(prefix=f"repolens_{repo_name}_")

    print(f"\n📦 Cloning {base_url} → {clone_dir}")
    _clone(url, clone_dir)

    print(f"🔍 Walking files...")
    source_files = _collect_files(clone_dir)
    print(f"   Found {len(source_files)} supported source files")

    all_chunks: list[CodeChunk] = []

    for file_path in tqdm(source_files, desc="Chunking", unit="file"):
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if len(source.encode("utf-8")) > MAX_FILE_BYTES:
            continue

        # Make path relative to the repo root for clean display + GitHub URLs
        rel_path = str(Path(file_path).relative_to(clone_dir))

        chunks = chunk_file(
            source=source,
            file_path=rel_path,
            repo_url=base_url,
            repo_root=clone_dir,
        )
        all_chunks.extend(chunks)

    print(f"✅ Extracted {len(all_chunks)} code chunks from {len(source_files)} files\n")
    return all_chunks, clone_dir


def _clone(url: str, target: str):
    """Shallow clone — only latest commit, no history."""
    if Path(target).exists() and any(Path(target).iterdir()):
        print(f"   (already cloned, reusing)")
        return
    Repo.clone_from(url, target, depth=1, single_branch=True)


def _collect_files(root: str) -> list[str]:
    """
    Walk the repo and return paths of all parseable source files,
    skipping vendor dirs, test files, and oversized files.
    """
    supported_exts = set(SUPPORTED_LANGUAGES.keys())
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in supported_exts:
                continue

            full_path = os.path.join(dirpath, fname)

            # Skip tiny files (likely empty or just comments)
            try:
                if os.path.getsize(full_path) < 50:
                    continue
            except OSError:
                continue

            results.append(full_path)

    return sorted(results)
