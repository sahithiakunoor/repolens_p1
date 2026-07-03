#!/usr/bin/env python3
"""
CLI: repolens-serve

Starts the RepoLens FastAPI server with uvicorn.

Usage:
    python -m repolens.scripts.serve
    repolens-serve --port 8080 --reload

Equivalent to running uvicorn directly, but reads host/port defaults from
settings so it stays consistent with the rest of the config.
"""

import argparse

from repolens.config import settings


def main():
    parser = argparse.ArgumentParser(description="Start the RepoLens API server.")
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=settings.port,
        help=f"Port to bind (default: {settings.port})",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    # Import uvicorn lazily so `repolens-serve --help` doesn't require it loaded.
    import uvicorn

    uvicorn.run(
        "repolens.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()