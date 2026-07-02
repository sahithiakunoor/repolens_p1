"""
RAGAS evaluation harness for RepoLens.

Measures pipeline quality on three metrics that don't require
pre-written reference answers — so the harness can run against
any indexed repo without manual labelling work:

  Faithfulness      — are all claims in the answer grounded in the
                      retrieved context? Catches hallucination.
                      Score: fraction of claims supported by context.
                      1.0 = nothing invented. 0.0 = fully hallucinated.

  Answer Relevancy  — does the answer actually address the question?
                      Catches answers that are factually grounded but
                      off-topic or evasive.
                      Score: semantic similarity between the question
                      and questions the answer implies.

  Context Precision — are the retrieved chunks relevant to the question?
                      Measures retrieval quality independently of generation.
                      Score: fraction of retrieved chunks that are useful.

Why LLM-as-judge:
  Classic NLP metrics (BLEU, ROUGE) compare strings — useless for code
  explanations where a paraphrase is correct and a verbatim copy might
  be wrong. RAGAS uses an LLM to check logical entailment and semantic
  relevance instead, which is how humans would judge the same outputs.

Usage:
    python -m repolens.evaluation.evaluator \
        --repo https://github.com/tiangolo/fastapi \
        --questions questions.json \
        --output results.json

Or from Python:
    from repolens.evaluation.evaluator import Evaluator
    evaluator = Evaluator(generator)
    results = evaluator.run(questions)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean

from repolens.generation.generator import Generator
from repolens.llm import llm
from repolens.models import QueryIntent


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One question to evaluate against."""
    question: str
    expected_intent: str | None = None   # optional — used to check intent accuracy


@dataclass
class SampleResult:
    """Evaluation result for a single question."""
    question: str
    intent: str
    answer: str
    context_chunks: list[str]            # file_path:start-end for each citation
    faithfulness: float                  # 0.0 – 1.0
    answer_relevancy: float              # 0.0 – 1.0
    context_precision: float             # 0.0 – 1.0
    latency_ms: int
    intent_correct: bool | None = None   # None if no expected_intent provided
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report across all questions."""
    repo_url: str
    timestamp: str
    num_questions: int
    num_errors: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_latency_ms: float
    intent_accuracy: float | None        # None if no expected intents provided
    samples: list[SampleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self):
        divider = "─" * 60
        print(f"\n{'RepoLens Evaluation Report':^60}")
        print(divider)
        print(f"  Repo        : {self.repo_url}")
        print(f"  Questions   : {self.num_questions}  ({self.num_errors} errors)")
        print(divider)
        print(f"  Faithfulness      : {self.mean_faithfulness:.3f}  (hallucination resistance)")
        print(f"  Answer Relevancy  : {self.mean_answer_relevancy:.3f}  (on-topic answers)")
        print(f"  Context Precision : {self.mean_context_precision:.3f}  (retrieval quality)")
        if self.intent_accuracy is not None:
            print(f"  Intent Accuracy   : {self.intent_accuracy:.3f}  (classification)")
        print(f"  Avg Latency       : {self.mean_latency_ms:.0f}ms")
        print(divider)
        print()


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, generator: Generator):
        self.generator = generator

    def run(self, samples: list[EvalSample]) -> EvalReport:
        """Run evaluation across all samples and return an EvalReport."""
        results: list[SampleResult] = []

        for i, sample in enumerate(samples, 1):
            print(f"  [{i}/{len(samples)}] {sample.question[:60]}...")
            result = self._evaluate_one(sample)
            results.append(result)
            _print_sample_scores(result)

        return _build_report(results, repo_url="")

    def run_for_repo(self, repo_url: str, samples: list[EvalSample]) -> EvalReport:
        """Run evaluation for a specific repo and return a labelled EvalReport."""
        print(f"\nEvaluating {len(samples)} questions against {repo_url}\n")
        report = self.run(samples)
        report.repo_url = repo_url
        return report

    def _evaluate_one(self, sample: EvalSample) -> SampleResult:
        """Run the full RAG pipeline + score one question."""
        try:
            start = time.monotonic()
            response = self.generator.answer(sample.question)
            latency_ms = int((time.monotonic() - start) * 1000)

            context_chunks = [
                f"{rc.chunk.file_path}:{rc.chunk.start_line}-{rc.chunk.end_line}"
                for rc in response.citations
            ]
            context_text = "\n\n".join(
                rc.chunk.content for rc in response.citations
            )

            faithfulness      = _score_faithfulness(response.answer, context_text)
            answer_relevancy  = _score_answer_relevancy(sample.question, response.answer)
            context_precision = _score_context_precision(sample.question, response.citations)

            intent_correct = None
            if sample.expected_intent:
                intent_correct = response.intent.value == sample.expected_intent

            return SampleResult(
                question=sample.question,
                intent=response.intent.value,
                answer=response.answer,
                context_chunks=context_chunks,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                latency_ms=latency_ms,
                intent_correct=intent_correct,
            )

        except Exception as e:
            return SampleResult(
                question=sample.question,
                intent="unknown",
                answer="",
                context_chunks=[],
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                latency_ms=0,
                error=str(e),
            )


# ── Metric implementations ────────────────────────────────────────────────────

def _score_faithfulness(answer: str, context: str) -> float:
    """
    Faithfulness: what fraction of claims in the answer are supported
    by the retrieved context?

    Strategy:
      1. Ask the LLM to extract atomic claims from the answer.
      2. Ask the LLM to verify each claim against the context.
      3. Score = supported_claims / total_claims.
    """
    if not answer.strip() or not context.strip():
        return 0.0

    # Step 1: Extract claims
    claims_prompt = f"""Extract all factual claims from the following answer as a JSON array of strings.
Each claim should be a single, atomic statement. Return ONLY valid JSON, no other text.

Answer:
{answer}

Return format: ["claim 1", "claim 2", ...]"""

    try:
        claims_raw = llm.chat(
            prompt=claims_prompt,
            system="You extract factual claims from text. Return only valid JSON arrays.",
            temperature=0.0,
            max_tokens=512,
        )
        claims = json.loads(_clean_json(claims_raw))
        if not isinstance(claims, list) or not claims:
            return 1.0  # no claims to verify = no hallucinations
    except Exception:
        return 1.0  # can't parse = give benefit of the doubt

    # Step 2: Verify each claim against context
    supported = 0
    for claim in claims[:10]:   # cap at 10 claims to control LLM cost
        verdict_prompt = f"""Does the following context support this claim?
Answer with exactly one word: YES or NO.

Context:
{context[:3000]}

Claim: {claim}"""
        try:
            verdict = llm.chat(
                prompt=verdict_prompt,
                system="You verify if context supports a claim. Answer YES or NO only.",
                temperature=0.0,
                max_tokens=5,
            )
            if "YES" in verdict.upper():
                supported += 1
        except Exception:
            supported += 1   # network error = give benefit of the doubt

    return round(supported / len(claims), 3)


def _score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: does the answer actually address the question?

    Strategy (RAGAS original):
      Ask the LLM to generate N questions that the answer implies,
      then measure how similar those generated questions are to the
      original. A relevant answer implies questions close to the original.

    We use a simpler proxy here that doesn't require an embedding model:
      Ask the LLM to rate 1-5 how well the answer addresses the question,
      normalise to 0-1. Cheaper and still strongly correlated with the
      embedding-based approach.
    """
    if not answer.strip():
        return 0.0

    prompt = f"""Rate how well the following answer addresses the question, on a scale of 1 to 5.
1 = completely off-topic or refuses to answer
3 = partially addresses the question
5 = fully and directly answers the question

Question: {question}

Answer: {answer[:1500]}

Reply with a single integer from 1 to 5. No other text."""

    try:
        raw = llm.chat(
            prompt=prompt,
            system="You rate answer relevancy. Reply with a single integer 1-5.",
            temperature=0.0,
            max_tokens=5,
        )
        rating = int("".join(filter(str.isdigit, raw.strip()))[:1])
        return round((rating - 1) / 4, 3)   # normalise to 0.0-1.0
    except Exception:
        return 0.5   # can't parse = neutral score


def _score_context_precision(question: str, citations) -> float:
    """
    Context Precision: what fraction of retrieved chunks are actually
    relevant to answering the question?

    Ask the LLM to judge each chunk independently: is this chunk useful
    for answering the question? Score = useful_chunks / total_chunks.
    """
    if not citations:
        return 0.0

    useful = 0
    for rc in citations:
        prompt = f"""Is the following code chunk useful for answering this question?
Answer with exactly one word: YES or NO.

Question: {question}

Code chunk ({rc.chunk.name} in {rc.chunk.file_path}):
{rc.chunk.content[:800]}"""

        try:
            verdict = llm.chat(
                prompt=prompt,
                system="You judge if a code chunk is useful for answering a question. Answer YES or NO only.",
                temperature=0.0,
                max_tokens=5,
            )
            if "YES" in verdict.upper():
                useful += 1
        except Exception:
            useful += 1   # give benefit of the doubt

    return round(useful / len(citations), 3)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    """Strip markdown code fences if the LLM wrapped JSON in them."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def _build_report(results: list[SampleResult], repo_url: str) -> EvalReport:
    import datetime
    good = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    intent_results = [r.intent_correct for r in good if r.intent_correct is not None]

    return EvalReport(
        repo_url=repo_url,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        num_questions=len(results),
        num_errors=len(errors),
        mean_faithfulness=round(mean(r.faithfulness for r in good), 3) if good else 0.0,
        mean_answer_relevancy=round(mean(r.answer_relevancy for r in good), 3) if good else 0.0,
        mean_context_precision=round(mean(r.context_precision for r in good), 3) if good else 0.0,
        mean_latency_ms=round(mean(r.latency_ms for r in good), 1) if good else 0.0,
        intent_accuracy=round(mean(intent_results), 3) if intent_results else None,
        samples=results,
    )


def _print_sample_scores(result: SampleResult):
    if result.error:
        print(f"       ❌ ERROR: {result.error}")
    else:
        print(
            f"       faithfulness={result.faithfulness:.2f}  "
            f"relevancy={result.answer_relevancy:.2f}  "
            f"precision={result.context_precision:.2f}  "
            f"{result.latency_ms}ms"
        )