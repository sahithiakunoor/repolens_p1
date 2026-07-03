"""
Evaluation runner: python -m repolens.evaluation

Loads a question set, runs it through the full RAG pipeline against an
already-indexed repo, scores each answer with the LLM-as-judge Evaluator,
prints a summary, and writes the report to JSON.

Prerequisite — index the repo first:
    repolens-ingest --repo https://github.com/tiangolo/fastapi

Then run the evaluation:
    python -m repolens.evaluation \
        --repo https://github.com/tiangolo/fastapi \
        --questions eval/fastapi_questions.json \
        --output eval/results.json

The questions file is a JSON array of objects with "question" and optional
"expected_intent":
    [
      {"question": "how does dependency injection work?", "expected_intent": "explain"},
      ...
    ]
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from repolens.config import settings
from repolens.generation.generator import Generator
from repolens.ingestion.indexer import Indexer
from repolens.evaluation.evaluator import Evaluator, EvalSample


def _load_questions(path: str) -> list[EvalSample]:
    """Load a JSON question file into EvalSample objects."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of question objects.")
    samples = []
    for i, item in enumerate(data):
        if "question" not in item:
            raise ValueError(f"Question {i} is missing the required 'question' field.")
        samples.append(EvalSample(
            question=item["question"],
            expected_intent=item.get("expected_intent"),
        ))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Run RepoLens evaluation against an indexed repo."
    )
    parser.add_argument(
        "--repo", required=True,
        help="GitHub repo URL (used to label the report)",
    )
    parser.add_argument(
        "--questions", required=True,
        help="Path to a JSON file of questions (e.g. eval/fastapi_questions.json)",
    )
    parser.add_argument(
        "--index-dir", default=str(settings.index_dir),
        help=f"Where the index is persisted (default: {settings.index_dir})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write the JSON report (e.g. eval/results.json)",
    )
    args = parser.parse_args()

    # 1. Load questions
    try:
        samples = _load_questions(args.questions)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Could not load questions from {args.questions}: {e}")
        sys.exit(1)
    logger.info(f"Loaded {len(samples)} questions from {args.questions}")

    # 2. Load the persisted index
    indexer = Indexer(persist_dir=args.index_dir)
    if indexer.count() == 0:
        logger.error(
            f"Index at {args.index_dir} is empty. Ingest the repo first:\n"
            f"  repolens-ingest --repo {args.repo}"
        )
        sys.exit(1)
    logger.info(f"Loaded index: {indexer.count()} chunks")

    # 3. Build the generator and evaluator
    generator = Generator(indexer=indexer)
    evaluator = Evaluator(generator=generator)

    # 4. Run
    report = evaluator.run_for_repo(repo_url=args.repo, samples=samples)

    # 5. Print summary
    report.print_summary()

    # 6. Persist
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
        logger.success(f"Report written to {args.output}")


if __name__ == "__main__":
    main()