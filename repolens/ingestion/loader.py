"""
Repo loader: clones any public GitHub repo and walks its files,
feeding each source file through the chunker.

Design decisions:
  - Shallow clone (depth=1) so we don't pull git history — fast and cheap
  - Detect the actual default branch post-clone via git so GitHub URLs
    are correct for repos that use 'master', 'develop', or anything else
  - Skip vendor dirs, test files, and auto-generated code by default
  - Progress bar via tqdm so the user knows it's working
  - Returns a flat list of CodeChunks ready for embedding
"""

import os
import subprocess
import tempfile
from pathlib import Path

from git import Repo
from loguru import logger
from tqdm import tqdm

from repolens.config import settings
from repolens.models import CodeChunk, SUPPORTED_LANGUAGES
from repolens.ingestion.chunker import chunk_file, detect_language

# Directories to always skip — noise, not signal
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".eggs", "*.egg-info",
    "vendor", "third_party", "fixtures", "migrations",
}


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

    logger.info(f"Cloning {base_url} → {clone_dir}")
    _clone(url, clone_dir)

    # Detect the actual default branch so GitHub blob URLs are always correct.
    # Falls back to 'main' if detection fails (e.g. bare/empty repos).
    default_branch = _detect_default_branch(clone_dir)
    logger.info(f"Default branch: {default_branch}")

    logger.info("Walking files...")
    source_files = _collect_files(clone_dir)
    logger.info(f"Found {len(source_files)} supported source files")

    all_chunks: list[CodeChunk] = []

    for file_path in tqdm(source_files, desc="Chunking", unit="file"):
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if len(source.encode("utf-8")) > settings.max_file_bytes:
            continue

        chunks = chunk_file(
            source=source,
            file_path=file_path,      # absolute path — chunker strips repo_root
            repo_url=base_url,
            repo_root=clone_dir,      # stripped by _relative_path() in chunker
            default_branch=default_branch,
        )
        all_chunks.extend(chunks)

    logger.info(f"Extracted {len(all_chunks)} code chunks from {len(source_files)} files")
    return all_chunks, clone_dir


def _clone(url: str, target: str):
    """Shallow clone — only latest commit, no history."""
    if Path(target).exists() and any(Path(target).iterdir()):
        logger.info("Already cloned, reusing")
        return
    Repo.clone_from(url, target, depth=1, single_branch=True)


def _detect_default_branch(clone_dir: str) -> str:
    """
    Detect the repo's default branch by asking git directly.

    'git rev-parse --abbrev-ref HEAD' returns the current branch name,
    which after a clone is always the remote's default branch.
    Falls back to 'main' if the command fails for any reason.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        branch = result.stdout.strip()
        if branch and branch != "HEAD":  # HEAD means detached — shouldn't happen after clone
            return branch
    except Exception as e:
        logger.warning(f"Could not detect default branch, falling back to 'main': {e}")
    return "main"


def _collect_files(root: str) -> list[str]:
    """
    Walk the repo and return absolute paths of all parseable source files,
    skipping vendor dirs and oversized files.
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