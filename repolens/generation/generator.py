"""
Generation layer — the piece that ties everything together.

Flow for a single query:
  1. Classify intent          (intent.py)
  2. Retrieve relevant chunks (retriever.py)
  3. Format chunks into context with file paths + GitHub links
  4. Build intent-aware prompt (prompts.py)
  5. Call the LLM             (llm.py)
  6. Return RAGResponse with answer + citations + intent + latency

The generator knows nothing about which LLM provider is active, which
retrieval model is running, or how chunks were indexed — it just
orchestrates the pieces via their public interfaces.
"""

import time

from repolens.generation.prompts import SYSTEM_PROMPT, build_prompt
from repolens.ingestion.indexer import Indexer
from repolens.llm import llm
from repolens.models import QueryIntent, RAGResponse, RetrievedChunk
from repolens.retrieval.intent import classify_intent
from repolens.retrieval.retriever import HybridRetriever


class Generator:
    def __init__(self, indexer: Indexer):
        self.retriever = HybridRetriever(indexer)

    def answer(self, query: str) -> RAGResponse:
        """
        Answer a natural-language question about the indexed repository.

        Returns a RAGResponse containing the answer text, the list of
        RetrievedChunks that grounded it (for citation display), the
        classified intent, and wall-clock latency in milliseconds.
        """
        start = time.monotonic()

        intent = classify_intent(query)
        chunks = self.retriever.retrieve(query)

        if not chunks:
            return RAGResponse(
                answer=(
                    "I couldn't find relevant code in the repository for that question. "
                    "Try rephrasing, or check that the repo was indexed successfully."
                ),
                citations=[],
                intent=intent,
                latency_ms=int((time.monotonic() - start) * 1000),
            )

        context = _format_context(chunks)
        prompt = build_prompt(query=query, context=context, intent=intent)
        answer_text = llm.chat(prompt=prompt, system=SYSTEM_PROMPT)

        return RAGResponse(
            answer=answer_text,
            citations=chunks,
            intent=intent,
            latency_ms=int((time.monotonic() - start) * 1000),
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Render retrieved chunks into a readable context block for the prompt.

    Format per chunk:
      ### src/some/file.py — MyClass.my_method (lines 42-67)
      GitHub: https://github.com/org/repo/blob/main/src/some/file.py#L42-L67
      ```python
      def my_method(self, ...):
          ...
      ```

    Leading with file path + name (not just raw code) gives the LLM enough
    context to produce accurate citations even when chunks look similar.
    """
    parts: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        chunk = rc.chunk
        header = (
            f"### [{i}] {chunk.file_path} — {chunk.name} "
            f"(lines {chunk.start_line}-{chunk.end_line})"
        )
        github_line = f"GitHub: {chunk.github_url}"
        code_block = f"```{chunk.language}\n{chunk.content}\n```"
        parts.append("\n".join([header, github_line, code_block]))

    return "\n\n".join(parts)