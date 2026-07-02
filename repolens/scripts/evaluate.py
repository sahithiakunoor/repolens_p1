#!/usr/bin/env python3
"""
RepoLens evaluation CLI — run RAGAS metrics against an indexed repo.

Usage:
    python -m repolens.scripts.evaluate \
        --repo https://github.com/tiangolo/fastapi \
        --questions eval/fastapi_questions.json \
        --output eval/results.json

Questions file format (JSON array):
    [
        {
            "question": "how does dependency injection work?",
            "expected_intent": "explain"
        },
        {
            "question": "where is the routing logic defined?",
            "expected_intent": "find"
        }
    ]

expected_intent is optional. Omit it to skip intent accuracy scoring.
"""

import argparse
import json
import sys
from pathlib import Path

from repolens.config import settings
from repolens.evaluation.evaluator import EvalSample, Evaluator
from repolens.generation.generator import Generator
from repolens.ingestion.indexer import Indexer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RepoLens pipeline quality using RAGAS metrics."
    )
    parser.add_argument("--repo", required=True, help="GitHub repo URL (must already be indexed)")
    parser.add_argument("--questions", required=True, help="Path to questions JSON file")
    parser.add_argument("--output", default=None, help="Path to save results JSON (optional)")
    parser.add_argument(
        "--index-dir", default=str(settings.index_dir),
        help=f"Index directory (default: {settings.index_dir})"
    )
    args = parser.parse_args()

    # ── Load questions ────────────────────────────────────────────────────────
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"❌  Questions file not found: {args.questions}")
        sys.exit(1)

    raw = json.loads(questions_path.read_text())
    samples = [
        EvalSample(
            question=q["question"],
            expected_intent=q.get("expected_intent"),
        )
        for q in raw
    ]
    print(f"✅  Loaded {len(samples)} evaluation questions")

    # ── Load index ────────────────────────────────────────────────────────────
    repo_slug = "__".join(args.repo.rstrip("/").split("/")[-2:])
    index_path = Path(args.index_dir) / repo_slug

    if not index_path.exists():
        print(f"❌  No index found at {index_path}")
        print(f"   Index the repo first: repolens-ingest --repo {args.repo}")
        sys.exit(1)

    print(f"✅  Loading index from {index_path}")
    indexer = Indexer(persist_dir=str(index_path))
    generator = Generator(indexer=indexer)

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = Evaluator(generator=generator)
    report = evaluator.run_for_repo(repo_url=args.repo, samples=samples)
    report.print_summary()

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"💾  Results saved to {args.output}\n")


if __name__ == "__main__":
    main()