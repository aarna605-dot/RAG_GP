"""The LLM that writes the final answer from retrieved passages.

Works with any provider that offers an OpenAI-style `/chat/completions` endpoint
(OpenAI, Groq, OpenRouter, Gemini's OpenAI-compatible endpoint, a local Ollama or vLLM, ...).
Configure it with environment variables; with none set, `llm_from_env()` returns None and
the RAG agents fall back to showing the best passages without a written answer.

    LLM_API_KEY    required
    LLM_MODEL      required, e.g. the model name from your provider's docs
    LLM_BASE_URL   optional, default https://api.openai.com/v1
    LLM_MAX_TOKENS optional, default 700
    LLM_PRICE_IN_PER_M / LLM_PRICE_OUT_PER_M   optional USD per million tokens (default 0, free tier)

Temperature is always 0, as the project brief requires for every setup.
"""
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.errors import AgentError


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> LLMResult: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 700,
        timeout: float = 90.0,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        self._max_retries = max_retries
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._transport = transport  # lets tests inject a fake server

    def complete(self, system: str, user: str) -> LLMResult:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                    resp = client.post(
                        f"{self._base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=payload,
                    )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    time.sleep(min(2 ** attempt * 2, 20))  # free tiers rate-limit often
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                usage = data.get("usage") or {}
                break
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:200].replace(self._api_key, "***")
                raise AgentError(f"The LLM request failed with status {exc.response.status_code}: {body}") from exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise AgentError(f"The LLM request failed: {type(exc).__name__}") from exc

        if not text:
            raise AgentError("The LLM returned an empty answer.")
        return LLMResult(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )


def llm_from_env() -> OpenAICompatibleLLM | None:
    api_key, model = os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
    if not (api_key and model):
        return None
    return OpenAICompatibleLLM(
        api_key=api_key,
        model=model,
        base_url=os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1",
        max_tokens=int(os.getenv("LLM_MAX_TOKENS") or 700),
    )
