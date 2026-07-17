"""
External model API clients: GeminiEmbedder (embeddings) and GroqClient
(generation). Grouped in one file because both are "call an external LLM
API with retries" and share the same resilience pattern - a reviewer
reading this file sees the entire external-dependency surface of the app.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from google import genai
from groq import Groq, RateLimitError, APIStatusError

from app.config import settings

logger = logging.getLogger("ai_clients")


async def _with_retry(fn, *, max_attempts: int = 5, base_delay: float = 1.0, label: str = "call"):
    """Shared exponential-backoff retry wrapper. Retries on rate limits and
    5xx errors; logs each attempt as structured JSON for observability."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised below
            is_retryable = isinstance(exc, RateLimitError) or (
                isinstance(exc, APIStatusError) and getattr(exc, "status_code", 500) >= 500
            ) or "429" in str(exc) or "rate" in str(exc).lower()

            log_payload = {
                "label": label,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "error": str(exc),
                "retryable": is_retryable,
            }
            logger.warning(json.dumps(log_payload))

            if not is_retryable or attempt >= max_attempts:
                raise

            delay = base_delay * (2 ** (attempt - 1))
            await asyncio.sleep(delay)


class GeminiEmbedder:
    """Wraps Gemini's embedding endpoint. Batches requests where possible
    and validates dimensionality/NaN issues are the caller's responsibility
    (checked in ingestion validation, not here, to keep this class a thin
    API wrapper)."""

    def __init__(self):
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def _call():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._client.models.embed_content(model=self._model, contents=texts),
            )

        result = await _with_retry(_call, label=f"gemini_embed_batch[{len(texts)}]")
        return [e.values for e in result.embeddings]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]


class GroqClient:
    """Wraps Groq's chat completion endpoint for the generation step.
    Prompt instructs the model to answer only from provided context and
    to cite filename + page/function for every claim."""

    # SYSTEM_PROMPT = (
    #     "You are an assistant answering questions using ONLY the provided context chunks. "
    #     "Each chunk is labeled with its source (filename and page number or function name). "
    #     "When you make a claim, cite the source it came from, e.g. '(source: report.pdf, page 3)' "
    #     "or '(source: utils.py, function: parse_data)'. "
    #     "If the context does not contain enough information to answer, say so explicitly rather than guessing."
    # )

    SYSTEM_PROMPT =  """
        You are a RAG assistant.
        Answer ONLY from the provided context.
        Write concise, natural responses.
        For document questions, summarize instead of quoting.
        For code questions, explain the logic rather than repeating implementation details unless explicitly asked.
        Avoid repeating the same information from multiple chunks.
        Return the answer as a single paragraph unless the question explicitly asks for steps or a list.
        Do not include raw newline characters (\n) in the response. Write the answer as natural prose.
        For code questions, describe the implementation exactly as written. Do not simplify APIs or algorithms. If the code uses a specific function (e.g., `random.choices()`), mention that function instead of assuming by yourself.
        If the context is insufficient, say:
        "I couldn't find that information in the uploaded documents."
        """

    def __init__(self):
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    import re

    def clean_chunk(text: str) -> str:
        # Remove extra whitespace from each line
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Join using a newline to preserve document structure
        return "\n".join(lines)
    async def generate_answer(self, query: str, context_chunks: list[dict]) -> str:

    
        parts = []

        for c in context_chunks:
            chunk_text = "\n".join(
                line.strip()
                for line in c["chunk_text"].splitlines()
                if line.strip()
            )

            parts.append(
                f"[Source: {c['filename']}"
                f"{', page ' + str(c['page_num']) if c.get('page_num') is not None else ''}"
                f"{', function ' + c['function_name'] if c.get('function_name') else ''}]\n"
                f"{chunk_text}"
            )

        context_block = "\n\n".join(parts)
        # user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"
        user_prompt = f"""
                        Question:
                        {query}
                        Context:
                        {context_block}
                        Answer naturally using only the context.
                        """

        async def _call():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                ),
            )

        response = await _with_retry(_call, label="groq_generate")
        return response.choices[0].message.content
