"""
Unified LLM client — one interface, five free/paid backends.

Usage:
    from repolens.llm import llm
    response = llm.chat("Explain this function: ...")

Switching providers requires only a .env change — no code changes anywhere else.

Provider quick-reference (all free unless noted):
  mistral  — DEFAULT. Codestral model, code-trained, ~1B tokens/month free.
             Sign up at console.mistral.ai, select Experiment plan.
  gemini   — Gemini 2.5 Flash, 1,500 req/day, 1M context window.
             Get key at aistudio.google.com.
  groq     — Llama 3.3 70B, 14,400 req/day, fastest inference.
             Get key at console.groq.com.
  ollama   — Fully local, no API key, needs `ollama serve` running.
  openai   — Paid. GPT-4o-mini. Best quality, costs money.

All providers share the same chat() interface — callers are provider-agnostic.
"""

import httpx
from loguru import logger
from repolens.config import settings


class LLMClient:
    """
    Routes chat() calls to whichever provider is set in LLM_PROVIDER env var.
    Returns a plain string — callers never see provider-specific response shapes.
    """

    def chat(
        self,
        prompt: str,
        system: str = "You are an expert code assistant. Be concise and precise.",
        temperature: float = 0.1,   # low = more deterministic, better for code
        max_tokens: int = 1024,
    ) -> str:
        provider = settings.llm_provider
        dispatch = {
            "mistral": self._mistral,
            "gemini":  self._gemini,
            "groq":    self._groq,
            "ollama":  self._ollama,
            "openai":  self._openai,
        }
        fn = dispatch.get(provider)
        if fn is None:
            raise ValueError(
                f"Unknown LLM_PROVIDER='{provider}'.\n"
                f"Valid options: {', '.join(dispatch.keys())}"
            )
        return fn(prompt, system, temperature, max_tokens)

    # ── Mistral (free, code-optimised) ── DEFAULT ─────────────────────────────

    def _mistral(self, prompt, system, temperature, max_tokens) -> str:
        """
        Mistral AI — free Experiment tier (~1B tokens/month).
        Uses Codestral by default: a model specifically trained on code corpora,
        making it significantly better than general models at explaining,
        generating, and reasoning about code.

        Models available on free tier:
          codestral-latest       ← best for RepoLens (code-specific)
          mistral-small-latest   ← good general fallback
          open-mistral-nemo      ← lightweight, very fast
        """
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.llm_model,   # default: codestral-latest
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise RuntimeError(
                    "Mistral API key invalid or missing.\n"
                    "Set MISTRAL_API_KEY in your .env file.\n"
                    "Get a free key at: https://console.mistral.ai"
                ) from e
            if e.response.status_code == 429:
                raise RuntimeError(
                    "Mistral rate limit hit. Wait a moment or switch provider:\n"
                    "  LLM_PROVIDER=groq in .env"
                ) from e
            raise

    # ── Gemini (free, large context window) ───────────────────────────────────

    def _gemini(self, prompt, system, temperature, max_tokens) -> str:
        """
        Google Gemini 2.5 Flash via AI Studio — free tier, 1,500 req/day.
        1M token context window is the standout feature — can pass
        entire source files without chunking concerns.

        Note: Gemini uses a different API shape (not OpenAI-compatible),
        so we call the REST endpoint directly.
        """
        model = settings.llm_model   # default: gemini-2.5-flash
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={settings.gemini_api_key}"
        )
        # Gemini merges system + user into a single "contents" array
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        try:
            resp = httpx.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return (
                resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RuntimeError(
                    "Gemini free tier daily limit reached (1,500 req/day).\n"
                    "Switch to Mistral: LLM_PROVIDER=mistral in .env"
                ) from e
            raise

    # ── Groq (free, fastest inference) ───────────────────────────────────────

    def _groq(self, prompt, system, temperature, max_tokens) -> str:
        """
        Groq LPU inference — fastest free option, sub-second responses.
        Free tier: 14,400 req/day on llama-3.1-8b-instant.
        OpenAI-compatible API.
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ── Ollama (local, no API key) ────────────────────────────────────────────

    def _ollama(self, prompt, system, temperature, max_tokens) -> str:
        """
        Local Ollama server — completely free, zero API calls, full privacy.
        Requires: `ollama serve` running + model pulled (`ollama pull llama3.2:3b`)
        """
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = httpx.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot connect to Ollama. Start it with:\n"
                "  ollama serve\n"
                "Then pull a model:\n"
                "  ollama pull llama3.2:3b"
            )

    # ── OpenAI (paid, optional) ───────────────────────────────────────────────

    def _openai(self, prompt, system, temperature, max_tokens) -> str:
        """OpenAI — paid. Only use if you have credits and need best quality."""
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        Verify the configured provider is reachable.
        Returns a dict with status and provider info — useful for the UI.
        """
        try:
            result = self.chat("Reply with exactly the word: READY", max_tokens=10)
            ok = "READY" in result.upper()
            return {
                "ok": ok,
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "response": result,
            }
        except Exception as e:
            logger.warning(f"LLM health check failed ({settings.llm_provider}): {e}")
            return {
                "ok": False,
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "error": str(e),
            }


# Singleton — import and use anywhere in the codebase
llm = LLMClient()
