"""
Intent classification: figures out *what kind* of question the user is asking
before we retrieve anything, because the right retrieval strategy and prompt
template differ by intent.

  EXPLAIN — "how does X work?"            → needs broader context, often a class + its callers
  EXAMPLE — "show me how to use X"        → needs usage sites, not just the definition
  FIND    — "where is X defined?"         → needs high-precision exact match, BM25-heavy
  DEBUG   — "why does X fail when..."     → needs the function + related error handling
  COMPARE — "difference between X and Y"  → needs to retrieve for *two* symbols, not one

Strategy: rule-based first, LLM fallback only when rules are ambiguous.

  - Rules are instant, free, deterministic, and unit-testable. They catch the
    large majority of real queries, which tend to use a handful of recurring
    phrasings ("how does", "show me", "where is", "why does", "difference between").
  - Rules alone don't generalize to phrasing we didn't anticipate, and fail
    silently (wrong intent, no error) rather than loudly. So when no rule
    clears a confidence margin over the others, we fall back to a single
    cheap LLM call rather than guessing.

This mirrors a common production pattern: fast deterministic path for the
common case, model fallback for the ambiguous tail.
"""

import re

from repolens.llm import llm
from repolens.models import QueryIntent

# ── Rule definitions ───────────────────────────────────────────────────────────
# Each intent has a list of regex patterns. More specific / less ambiguous
# patterns are listed first within each intent (not load-bearing for scoring,
# just readability).

_PATTERNS: dict[QueryIntent, list[re.Pattern]] = {
    QueryIntent.COMPARE: [
        re.compile(r"\bdifference between\b"),
        re.compile(r"\bcompare(d)?\b"),
        re.compile(r"\bvs\.?\b"),
        re.compile(r"\bversus\b"),
        re.compile(r"\bwhich (one|is better)\b"),
        re.compile(r"\bbetter\b.*\bthan\b"),
    ],
    QueryIntent.DEBUG: [
        re.compile(r"\bwhy (does|is|do|am i)\b.*\b(fail|error|break|crash|wrong)\b"),
        re.compile(r"\b(error|exception|traceback|stack trace)\b"),
        re.compile(r"\bnot working\b"),
        re.compile(r"\bbug\b"),
        re.compile(r"\bfix(ing)?\b"),
        re.compile(r"\bfails? when\b"),
        re.compile(r"\bthrows?\b"),
    ],
    QueryIntent.FIND: [
        re.compile(r"\bwhere is\b"),
        re.compile(r"\bwhere(\'s| is| are)\b.*\bdefined\b"),
        re.compile(r"\bwhich file\b"),
        re.compile(r"\bfind\b.*\b(function|class|method|definition)\b"),
        re.compile(r"\blocate\b"),
    ],
    QueryIntent.EXAMPLE: [
        re.compile(r"\bshow me\b.*\b(how to|example|usage)\b"),
        re.compile(r"\bexample(s)? of\b"),
        re.compile(r"\bhow (do|can) i use\b"),
        re.compile(r"\busage of\b"),
        re.compile(r"\bsample code\b"),
    ],
    QueryIntent.EXPLAIN: [
        re.compile(r"\bhow does\b.*\bwork\b"),
        re.compile(r"\bexplain\b"),
        re.compile(r"\bwhat (does|is)\b"),
        re.compile(r"\bwalk me through\b"),
        re.compile(r"\bpurpose of\b"),
    ],
}

# Minimum margin (in matched-pattern count) the top intent needs over the
# runner-up before we trust the rule-based result outright.
_CONFIDENCE_MARGIN = 1

_FALLBACK_SYSTEM_PROMPT = (
    "You are an intent classifier for a code question-answering system. "
    "Given a user's question about a codebase, classify it into exactly one "
    "of these categories:\n"
    "  explain - asking how something works or what it does\n"
    "  example - asking for usage examples or how to use something\n"
    "  find    - asking where something is defined or located\n"
    "  debug   - asking why something fails or has a bug\n"
    "  compare - asking about differences between two things\n"
    "Reply with exactly one word: explain, example, find, debug, or compare. "
    "No punctuation, no explanation."
)


def classify_intent(query: str) -> QueryIntent:
    """
    Classify a user query into a QueryIntent.

    Tries rule-based matching first. Falls back to an LLM call only when
    no intent clears a clear confidence margin over the runner-up — keeping
    the common case fast and free, and the ambiguous tail accurate.
    """
    rule_result = _classify_by_rules(query)
    if rule_result is not None:
        return rule_result
    return _classify_by_llm(query)


def _classify_by_rules(query: str) -> QueryIntent | None:
    """Score each intent by number of matched patterns. Return the winner
    only if it clears _CONFIDENCE_MARGIN over the runner-up; otherwise None
    signals the caller to fall back to the LLM."""
    query_lower = query.lower()
    scores: dict[QueryIntent, int] = {
        intent: sum(1 for pattern in patterns if pattern.search(query_lower))
        for intent, patterns in _PATTERNS.items()
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score == 0:
        return None  # nothing matched at all — ambiguous, defer to LLM
    if top_score - runner_up_score < _CONFIDENCE_MARGIN:
        return None  # too close to call — defer to LLM
    return top_intent


def _classify_by_llm(query: str) -> QueryIntent:
    """LLM fallback for queries the rules couldn't confidently classify."""
    try:
        raw = llm.chat(
            prompt=query,
            system=_FALLBACK_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=10,
        )
        cleaned = raw.strip().lower().strip(".")
        for intent in QueryIntent:
            if intent.value == cleaned:
                return intent
    except Exception:
        pass
    # If the LLM call fails or returns something unparseable, default to
    # EXPLAIN — the broadest, most forgiving retrieval strategy, rather than
    # raising and breaking the whole query.
    return QueryIntent.EXPLAIN