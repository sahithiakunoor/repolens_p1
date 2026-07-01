"""
Intent-aware prompt templates for the generation layer.

Why separate from generator.py:
  Prompt wording is tuned independently of retrieval/generation logic.
  Keeping them here means a prompt change is a one-file edit, not a
  reason to touch the generator. It also makes it easy to A/B test
  different prompt wordings without touching any control flow.

Each template receives the same two variables:
  {context}  — the retrieved code chunks, pre-formatted with file paths
                and GitHub links
  {query}    — the user's original question, verbatim

The system prompt establishes the assistant's role and citation rules.
The user prompt varies by intent to steer the LLM toward the right
kind of answer: narrative explanation vs. usage example vs. exact
location vs. failure analysis vs. side-by-side comparison.
"""

from repolens.models import QueryIntent


SYSTEM_PROMPT = """You are RepoLens, an expert code assistant that answers \
questions about a specific GitHub repository. You have been given relevant \
code chunks retrieved from the repo.

Rules:
- Base your answer ONLY on the provided code chunks. Do not invent code.
- Always cite the source of any code you reference using the format:
  [FileName:StartLine-EndLine](github_url)
- If the chunks don't contain enough information to fully answer the question,
  say so clearly rather than guessing.
- Be concise and precise. Developers value accuracy over length.
"""

_TEMPLATES: dict[QueryIntent, str] = {

    QueryIntent.EXPLAIN: """\
Here are relevant code chunks from the repository:

{context}

Question: {query}

Explain how this works. Walk through the logic step by step, covering:
- What it does at a high level
- Key implementation details
- How the pieces fit together

Cite each chunk you reference with [FileName:Lines](url).""",

    QueryIntent.EXAMPLE: """\
Here are relevant code chunks from the repository:

{context}

Question: {query}

Show how to use this in practice. Your answer should include:
- A concrete usage example drawn from the retrieved code
- What parameters or arguments are expected
- What the caller gets back

Cite each chunk you reference with [FileName:Lines](url).""",

    QueryIntent.FIND: """\
Here are relevant code chunks from the repository:

{context}

Question: {query}

Identify exactly where this is defined or implemented:
- The file path
- The line range
- A one-sentence summary of what it does

Cite with [FileName:Lines](url). If multiple definitions exist, list all of them.""",

    QueryIntent.DEBUG: """\
Here are relevant code chunks from the repository:

{context}

Question: {query}

Analyze what could cause this issue:
- What the relevant code is doing
- Where and why it might fail given the described conditions
- What a fix or workaround might look like

Cite each chunk you reference with [FileName:Lines](url).""",

    QueryIntent.COMPARE: """\
Here are relevant code chunks from the repository:

{context}

Question: {query}

Compare the two, covering:
- What each one does
- Key differences in behaviour, interface, or use case
- When you would use one over the other

Cite each chunk you reference with [FileName:Lines](url).""",
}


def build_prompt(query: str, context: str, intent: QueryIntent) -> str:
    """Return the filled user prompt for the given intent."""
    template = _TEMPLATES[intent]
    return template.format(query=query, context=context)