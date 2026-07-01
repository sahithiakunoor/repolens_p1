"""
Unit tests for repolens.generation

Covers:
  - prompts.py: correct template selected per intent, variables substituted
  - generator.py: intent classification wired correctly, context formatted
    properly, LLM called with the right prompt shape, RAGResponse assembled
    correctly, empty-retrieval case handled gracefully

Everything that makes a real network call (LLM, retriever) is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from repolens.generation.prompts import SYSTEM_PROMPT, build_prompt
from repolens.generation.generator import Generator, _format_context
from repolens.models import CodeChunk, QueryIntent, RAGResponse, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_chunk(chunk_id="chunk_1", name="my_func", file_path="src/app.py",
               start_line=10, end_line=20, language="python") -> RetrievedChunk:
    chunk = CodeChunk(
        chunk_id=chunk_id,
        content="def my_func():\n    return 42",
        chunk_type="function",
        name=name,
        file_path=file_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        docstring="Returns 42.",
        parent_class="",
        imports=[],
        repo_url="https://github.com/org/repo",
        github_url=f"https://github.com/org/repo/blob/main/{file_path}#L{start_line}-L{end_line}",
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


@pytest.fixture
def mock_generator():
    """Generator with a mocked indexer — never loads real ChromaDB or models."""
    with patch("repolens.generation.generator.HybridRetriever") as mock_retriever_cls:
        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever
        gen = Generator(indexer=MagicMock())
        gen.retriever = mock_retriever
        return gen


# ── prompts.py ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("intent,expected_keyword", [
    (QueryIntent.EXPLAIN,  "step by step"),
    (QueryIntent.EXAMPLE,  "usage example"),
    (QueryIntent.FIND,     "line range"),
    (QueryIntent.DEBUG,    "cause this issue"),
    (QueryIntent.COMPARE,  "Key differences"),
])
def test_build_prompt_selects_correct_template(intent, expected_keyword):
    prompt = build_prompt(
        query="test query",
        context="<some context>",
        intent=intent,
    )
    assert expected_keyword in prompt


def test_build_prompt_injects_query_and_context():
    prompt = build_prompt(
        query="how does retry work?",
        context="### [1] src/retry.py",
        intent=QueryIntent.EXPLAIN,
    )
    assert "how does retry work?" in prompt
    assert "### [1] src/retry.py" in prompt


def test_system_prompt_contains_citation_rule():
    # Ensures the LLM will always be instructed to cite sources.
    assert "[FileName:StartLine-EndLine]" in SYSTEM_PROMPT
    assert "ONLY on the provided code chunks" in SYSTEM_PROMPT


# ── _format_context ───────────────────────────────────────────────────────────

def test_format_context_includes_file_path_and_name():
    rc = make_chunk(file_path="src/app.py", name="my_func")
    context = _format_context([rc])
    assert "src/app.py" in context
    assert "my_func" in context


def test_format_context_includes_github_url():
    rc = make_chunk(start_line=10, end_line=20)
    context = _format_context([rc])
    assert "github.com" in context
    assert "L10" in context


def test_format_context_includes_code_block():
    rc = make_chunk()
    context = _format_context([rc])
    assert "```python" in context
    assert "def my_func" in context


def test_format_context_numbers_chunks():
    chunks = [make_chunk(chunk_id=str(i), name=f"func_{i}") for i in range(3)]
    context = _format_context(chunks)
    assert "[1]" in context
    assert "[2]" in context
    assert "[3]" in context


def test_format_context_empty_list():
    assert _format_context([]) == ""


# ── Generator.answer() ────────────────────────────────────────────────────────

def test_answer_returns_rag_response(mock_generator):
    mock_generator.retriever.retrieve.return_value = [make_chunk()]
    with patch("repolens.generation.generator.llm") as mock_llm:
        mock_llm.chat.return_value = "Here is the explanation."
        with patch("repolens.generation.generator.classify_intent",
                   return_value=QueryIntent.EXPLAIN):
            result = mock_generator.answer("how does my_func work?")

    assert isinstance(result, RAGResponse)
    assert result.answer == "Here is the explanation."
    assert result.intent == QueryIntent.EXPLAIN
    assert len(result.citations) == 1
    assert result.latency_ms >= 0


def test_answer_passes_system_prompt_to_llm(mock_generator):
    mock_generator.retriever.retrieve.return_value = [make_chunk()]
    with patch("repolens.generation.generator.llm") as mock_llm:
        mock_llm.chat.return_value = "Answer."
        with patch("repolens.generation.generator.classify_intent",
                   return_value=QueryIntent.EXPLAIN):
            mock_generator.answer("how does this work?")

        _, kwargs = mock_llm.chat.call_args
        assert kwargs["system"] == SYSTEM_PROMPT


def test_answer_passes_correct_intent_prompt_to_llm(mock_generator):
    mock_generator.retriever.retrieve.return_value = [make_chunk()]
    with patch("repolens.generation.generator.llm") as mock_llm:
        mock_llm.chat.return_value = "Answer."
        with patch("repolens.generation.generator.classify_intent",
                   return_value=QueryIntent.FIND):
            mock_generator.answer("where is my_func?")

        _, kwargs = mock_llm.chat.call_args
        # FIND template should mention "line range"
        assert "line range" in kwargs["prompt"]


def test_answer_handles_empty_retrieval_gracefully(mock_generator):
    mock_generator.retriever.retrieve.return_value = []
    with patch("repolens.generation.generator.classify_intent",
               return_value=QueryIntent.EXPLAIN):
        result = mock_generator.answer("some query")

    assert isinstance(result, RAGResponse)
    assert result.citations == []
    assert "couldn't find" in result.answer.lower()


@pytest.mark.parametrize("intent", list(QueryIntent))
def test_answer_propagates_intent_for_all_intents(mock_generator, intent):
    mock_generator.retriever.retrieve.return_value = [make_chunk()]
    with patch("repolens.generation.generator.llm") as mock_llm:
        mock_llm.chat.return_value = "Answer."
        with patch("repolens.generation.generator.classify_intent",
                   return_value=intent):
            result = mock_generator.answer("any query")
    assert result.intent == intent