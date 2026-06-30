"""
CLI: repolens-ingest

Usage:
    python -m repolens.scripts.ingest --repo https://github.com/tiangolo/fastapi
    repolens-ingest --repo https://github.com/langchain-ai/langchain --index-dir ./my_index
"""

import argparse
import sys
import time

from loguru import logger

from repolens.ingestion.loader import load_repo
from repolens.ingestion.indexer import Indexer
from repolens.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a GitHub repository into the RepoLens index."
    )
    parser.add_argument(
        "--repo", required=True,
        help="Public GitHub URL e.g. https://github.com/tiangolo/fastapi"
    )
    parser.add_argument(
        "--index-dir", default=str(settings.index_dir),
        help=f"Where to persist the index (default: {settings.index_dir})"
    )
    parser.add_argument(
        "--clone-dir", default=str(settings.clone_dir),
        help="Where to clone the repo (default: temp dir)"
    )
    args = parser.parse_args()

    logger.info(f"Starting ingestion for: {args.repo}")
    t0 = time.time()

    # Step 1: Clone and chunk
    chunks, clone_path = load_repo(
        github_url=args.repo,
        clone_dir=args.clone_dir if args.clone_dir else None,
    )

    if not chunks:
        logger.error("No chunks extracted. Check the repo URL and supported languages.")
        sys.exit(1)

    # Step 2: Embed and index
    indexer = Indexer(persist_dir=args.index_dir)
    indexer.index(chunks)

    elapsed = time.time() - t0
    logger.success(
        f"Done in {elapsed:.1f}s — "
        f"{len(chunks)} chunks indexed into {args.index_dir}"
    )


if __name__ == "__main__":
    main()
