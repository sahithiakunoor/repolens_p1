#!/usr/bin/env python3
"""
RepoLens CLI — query the API and pretty-print the response.

Usage:
    python -m repolens.scripts.query \
        --repo https://github.com/tiangolo/fastapi \
        --question "how does dependency injection work?"

Or after pip install -e .:
    repolens-query --repo <url> --question "<question>"
"""

import argparse
import json
import sys

import httpx


# ── ANSI colours (work on macOS/Linux terminals) ──────────────────────────────
BOLD    = "\033[1m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
GREY    = "\033[90m"
RESET   = "\033[0m"
DIVIDER = "─" * 70


def main():
    parser = argparse.ArgumentParser(
        description="Query the RepoLens API and pretty-print the response."
    )
    parser.add_argument("--repo", required=True, help="GitHub repo URL")
    parser.add_argument("--question", required=True, help="Your question")
    parser.add_argument(
        "--host", default="http://localhost:8000", help="API host (default: localhost:8000)"
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}RepoLens{RESET}")
    print(DIVIDER)
    print(f"{BOLD}Repo    :{RESET} {args.repo}")
    print(f"{BOLD}Question:{RESET} {args.question}")
    print(DIVIDER)
    print(f"{GREY}Querying...{RESET}\n")

    try:
        response = httpx.post(
            f"{args.host}/query",
            json={"repo_url": args.repo, "question": args.question},
            timeout=60,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        print(f"❌  Cannot connect to {args.host}")
        print(f"   Start the server with: uvicorn repolens.api:app --port 8000")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", e.response.text)
        print(f"❌  API error {e.response.status_code}: {detail}")
        sys.exit(1)

    data = response.json()

    # ── Intent ────────────────────────────────────────────────────────────────
    intent_emoji = {
        "explain": "💡", "example": "📝",
        "find": "🔍", "debug": "🐛", "compare": "⚖️",
    }
    intent = data.get("intent", "explain")
    emoji  = intent_emoji.get(intent, "💡")
    print(f"{GREY}Intent detected:{RESET} {emoji}  {intent.upper()}")
    print()

    # ── Answer ────────────────────────────────────────────────────────────────
    print(f"{BOLD}{GREEN}Answer{RESET}")
    print(DIVIDER)
    print(data["answer"])
    print()

    # ── Citations ─────────────────────────────────────────────────────────────
    citations = data.get("citations", [])
    if citations:
        print(f"{BOLD}{YELLOW}Sources ({len(citations)}){RESET}")
        print(DIVIDER)
        for i, c in enumerate(citations, 1):
            print(f"  {BOLD}[{i}]{RESET} {c['name']}")
            print(f"       {GREY}{c['file_path']}  L{c['start_line']}-{c['end_line']}{RESET}")
            print(f"       {CYAN}{c['github_url']}{RESET}")
            print()

    # ── Latency ───────────────────────────────────────────────────────────────
    latency = data.get("latency_ms", 0)
    print(f"{GREY}⏱  {latency}ms{RESET}\n")


if __name__ == "__main__":
    main()