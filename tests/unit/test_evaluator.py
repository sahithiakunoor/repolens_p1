"""
Unit tests for repolens.evaluation.evaluator

All LLM calls are mocked — we test the scoring logic, report
aggregation, and error handling, not the LLM's judgment itself.
"""

from unittest.mock import MagicMock, patch

import pytest

from repolens.evaluation.evaluator import (
    EvalSample,
    Evaluator,
    SampleResult,
    _clean_json,
    _score_answer_relevancy,
    _score_context_precision,
    _score_faithfulness,
    _build_report,
)
from repolens.models import CodeChunk, QueryIntent, RAGResponse, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_retrieved_chunk(name="func", file_path="src/app.py") -> RetrievedChunk:
    chunk = CodeChunk(
        chunk_id="c1", content="def func(): return 42",
        chunk_type="function", name=name, file_path=file_path,
        language="python", start_line=1, end_line=5,
        docstring="", parent_class="", imports=[],
        repo_url="https://github.com/org/repo",
        github_url="https://github.com/org/repo/blob/main/src/app.py#L1-L5",
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


def make_rag_response(intent=QueryIntent.EXPLAIN) -> RAGResponse:
    return RAGResponse(
        answer="The function returns 42.",
        citations=[make_retrieved_chunk()],
        intent=intent,
        latency_ms=100,
    )


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.answer.return_value = make_rag_response()
    return gen


@pytest.fixture
def evaluator(mock_generator):
    return Evaluator(generator=mock_generator)


# ── _clean_json ───────────────────────────────────────────────────────────────

def test_clean_json_strips_markdown_fences():
    raw = '```json\n["claim 1", "claim 2"]\n```'
    assert _clean_json(raw) == '["claim 1", "claim 2"]'


def test_clean_json_passthrough_clean_json():
    raw = '["claim 1"]'
    assert _clean_json(raw) == '["claim 1"]'


# ── _score_faithfulness ───────────────────────────────────────────────────────

def test_faithfulness_all_supported():
    with patch("repolens.evaluation.evaluator.llm") as mock_llm:
        mock_llm.chat.side_effect = ['["The function returns 42."]', "YES"]
        score = _score_faithfulness("The function returns 42.", "def func(): return 42")
    assert score == 1.0


def test_faithfulness_none_supported():
    with patch("repolens.evaluation.evaluator.llm") as mock_llm:
        mock_llm.chat.side_effect = ['["The function returns 100."]', "NO"]
        score = _score_faithfulness("The function returns 100.", "def func(): return 42")
    assert score == 0.0


def test_faithfulness_empty_answer_returns_zero():
    score = _score_faithfulness("", "some context")
    assert score == 0.0


def test_faithfulness_llm_parse_failure_returns_one():
    with patch("repolens.evaluation.evaluator.llm") as mock_llm:
        mock_llm.chat.return_value = "not valid json at all"
        score = _score_faithfulness("some answer", "some context")
    assert score == 1.0


# ── _score_answer_relevancy ───────────────────────────────────────────────────

@pytest.mark.parametrize("rating,expected", [
    ("5", 1.0),
    ("3", 0.5),
    ("1", 0.0),
])
def test_answer_relevancy_normalises_rating(rating, expected):
    with patch("repolens.evaluation.evaluator.llm") as mock_llm:
        mock_llm.chat.return_value = rating
        score = _score_answer_relevancy("how does X work?", "X works by doing Y.")
    assert score == expected


def test_answer_relevancy_empty_answer_returns_zero():
    score = _score_answer_relevancy("question", "")
    assert score == 0.0


def test_answer_relevancy_llm_failure_returns_neutral():
    with patch("repolens.evaluation.evaluator.llm") as mock_llm:
        mock_llm.chat.side_effect = RuntimeError("provider down")
        score = _score_answer_relevancy("question", "answer")
    assert score == 0.5


# ── _score_context_precision ──────────────────────────────────────────────────

def test_context_precision_all_useful():
    citations = [make_retrieved_chunk(), make_retrieved_chunk()]
    with patch("repolens.evaluation.evaluator.llm") as mock_llm:
        mock_llm.chat.return_value = "YES"
        score = _score_context_precision("how does func work?", citations)
    assert score == 1.0


def test_context_precision_none_useful():
    citations = [make_retrieved_chunk()]
    with patch("repolens.evaluation.evaluator.llm") as mock_llm:
        mock_llm.chat.return_value = "NO"
        score = _score_context_precision("how does func work?", citations)
    assert score == 0.0


def test_context_precision_empty_citations_returns_zero():
    score = _score_context_precision("question", [])
    assert score == 0.0


# ── Evaluator._evaluate_one ───────────────────────────────────────────────────

def test_evaluate_one_returns_sample_result(evaluator):
    with patch("repolens.evaluation.evaluator._score_faithfulness", return_value=0.9), \
         patch("repolens.evaluation.evaluator._score_answer_relevancy", return_value=0.8), \
         patch("repolens.evaluation.evaluator._score_context_precision", return_value=0.7):
        result = evaluator._evaluate_one(EvalSample(question="how does X work?"))

    assert isinstance(result, SampleResult)
    assert result.faithfulness == 0.9
    assert result.answer_relevancy == 0.8
    assert result.context_precision == 0.7
    assert result.error is None


def test_evaluate_one_checks_intent_when_expected(evaluator):
    with patch("repolens.evaluation.evaluator._score_faithfulness", return_value=1.0), \
         patch("repolens.evaluation.evaluator._score_answer_relevancy", return_value=1.0), \
         patch("repolens.evaluation.evaluator._score_context_precision", return_value=1.0):
        result = evaluator._evaluate_one(
            EvalSample(question="how does X work?", expected_intent="explain")
        )
    assert result.intent_correct is True


def test_evaluate_one_captures_generator_errors(evaluator):
    evaluator.generator.answer.side_effect = RuntimeError("LLM timeout")
    result = evaluator._evaluate_one(EvalSample(question="question"))
    assert result.error == "LLM timeout"
    assert result.faithfulness == 0.0


# ── _build_report ─────────────────────────────────────────────────────────────

def test_build_report_aggregates_correctly():
    results = [
        SampleResult("q1", "explain", "ans", [], 0.9, 0.8, 0.7, 100, True),
        SampleResult("q2", "find",    "ans", [], 0.8, 0.9, 0.6, 200, False),
    ]
    report = _build_report(results, repo_url="https://github.com/org/repo")
    assert report.mean_faithfulness == pytest.approx(0.85, abs=0.01)
    assert report.mean_answer_relevancy == pytest.approx(0.85, abs=0.01)
    assert report.mean_context_precision == pytest.approx(0.65, abs=0.01)
    assert report.intent_accuracy == pytest.approx(0.5, abs=0.01)
    assert report.num_errors == 0


def test_build_report_excludes_errored_samples():
    results = [
        SampleResult("q1", "explain", "ans", [], 0.9, 0.8, 0.7, 100),
        SampleResult("q2", "unknown", "",    [], 0.0, 0.0, 0.0, 0, error="LLM timeout"),
    ]
    report = _build_report(results, repo_url="")
    assert report.num_errors == 1
    assert report.mean_faithfulness == 0.9   # only the good sample counted