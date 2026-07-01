"""
Unit tests for repolens.retrieval.intent

Covers:
  - Rule-based classification for clear, unambiguous queries (one per intent).
  - Ambiguous queries correctly falling through to the LLM fallback.
  - LLM fallback parsing (mocked — no real API calls).
  - LLM fallback failure defaulting safely to EXPLAIN.
"""

from unittest.mock import patch

import pytest

from repolens.models import QueryIntent
from repolens.retrieval.intent import _classify_by_rules, classify_intent


# ── Rule-based: one clear example per intent ──────────────────────────────────

@pytest.mark.parametrize(
    "query,expected",
    [
        ("how does the authentication middleware work?", QueryIntent.EXPLAIN),
        ("show me an example of using BaseTool", QueryIntent.EXAMPLE),
        ("where is the retry logic defined?", QueryIntent.FIND),
        ("why does this throw a KeyError when the cache is empty?", QueryIntent.DEBUG),
        ("difference between RunnableSequence and RunnableParallel", QueryIntent.COMPARE),
        ("what is this class for", QueryIntent.EXPLAIN),
        ("explain the retry decorator", QueryIntent.EXPLAIN),
        ("sample code for the HTTP client", QueryIntent.EXAMPLE),
        ("locate the config loader", QueryIntent.FIND),
        ("this is not working, why?", QueryIntent.DEBUG),
        ("RunnableSequence vs RunnableParallel", QueryIntent.COMPARE),
    ],
)
def test_classify_by_rules_clear_cases(query, expected):
    assert _classify_by_rules(query) == expected


# ── Rule-based: ambiguous queries should defer (return None) ─────────────────

@pytest.mark.parametrize(
    "query",
    [
        "why does the team use a factory pattern here",  # "why" but semantically EXPLAIN
        "tell me about the cache module",                 # no strong keyword at all
        "",                                                # empty query
    ],
)
def test_classify_by_rules_ambiguous_defers(query):
    assert _classify_by_rules(query) is None


# ── classify_intent(): rule path should short-circuit, never touching the LLM ─

def test_classify_intent_uses_rules_without_calling_llm():
    with patch("repolens.retrieval.intent.llm") as mock_llm:
        result = classify_intent("where is the retry logic defined?")
        assert result == QueryIntent.FIND
        mock_llm.chat.assert_not_called()


# ── classify_intent(): ambiguous query routes to the (mocked) LLM fallback ───

@pytest.mark.parametrize(
    "llm_response,expected",
    [
        ("explain", QueryIntent.EXPLAIN),
        ("EXAMPLE", QueryIntent.EXAMPLE),
        ("find.", QueryIntent.FIND),       # trailing punctuation should be stripped
        (" debug ", QueryIntent.DEBUG),    # whitespace should be stripped
        ("compare", QueryIntent.COMPARE),
    ],
)
def test_classify_intent_llm_fallback_parses_response(llm_response, expected):
    with patch("repolens.retrieval.intent.llm") as mock_llm:
        mock_llm.chat.return_value = llm_response
        result = classify_intent("tell me about the cache module")
        assert result == expected
        mock_llm.chat.assert_called_once()


# ── classify_intent(): unparseable or failing LLM response defaults safely ───

def test_classify_intent_llm_fallback_unparseable_defaults_to_explain():
    with patch("repolens.retrieval.intent.llm") as mock_llm:
        mock_llm.chat.return_value = "I'm not sure, maybe something else entirely"
        result = classify_intent("tell me about the cache module")
        assert result == QueryIntent.EXPLAIN


def test_classify_intent_llm_fallback_exception_defaults_to_explain():
    with patch("repolens.retrieval.intent.llm") as mock_llm:
        mock_llm.chat.side_effect = RuntimeError("provider unreachable")
        result = classify_intent("tell me about the cache module")
        assert result == QueryIntent.EXPLAIN