"""Minimal LLM interface so R3 can be built and tested against mocks.

R4/R7 may already have a client; if so, wrap it to satisfy `LLMClient` and delete this file.
Contract: temperature 0 (project rule), returns text + token counts so cost is always tracked.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Union

from .schemas import Usage


@dataclass
class LLMResponse:
    text: str
    usage: Usage


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse: ...


class MockLLM:
    """Deterministic stand-in. `responses` is a list (consumed in order, last one repeats)
    or a callable (system, user) -> str. Token counts are a rough chars/4 estimate."""

    def __init__(self, responses: Union[List[str], Callable[[str, str], str]]):
        self.responses = responses
        self.calls: List[tuple] = []

    def complete(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse:
        self.calls.append((system, user))
        if callable(self.responses):
            text = self.responses(system, user)
        else:
            i = min(len(self.calls) - 1, len(self.responses) - 1)
            text = self.responses[i]
        return LLMResponse(text, Usage(1, (len(system) + len(user)) // 4, len(text) // 4, 0.0))


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Chat-completions client for any OpenAI-compatible endpoint (Gemini, Groq, Ollama, vLLM...).
    Env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, optional LLM_REASONING_EFFORT (e.g. "low"/"none").
    Stdlib only. Retries 429/5xx with backoff (free tiers rate-limit constantly) and shows the
    server's error body instead of a bare "HTTP 400".
    NOTE: tested against a local fake server only, not a live provider."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 60.0, max_retries: int = 4,
                 reasoning_effort: Optional[str] = None):
        self.base_url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ["LLM_MODEL"]
        self.timeout, self.max_retries = timeout, max_retries
        self.reasoning_effort = reasoning_effort or os.environ.get("LLM_REASONING_EFFORT")

    def complete(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.reasoning_effort:   # thinking models can spend max_tokens on hidden reasoning
            payload["reasoning_effort"] = self.reasoning_effort
        body = json.dumps(payload).encode()

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    wait = float(e.headers.get("Retry-After") or 0) or min(2 ** attempt * 2, 30)
                    time.sleep(wait)
                    continue
                raise LLMError(f"HTTP {e.code} from {self.base_url}: {detail}") from None
            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise LLMError(f"cannot reach {self.base_url}: {e.reason}") from None

        ms = (time.perf_counter() - t0) * 1000
        u = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text and choice.get("finish_reason") == "length":
            raise LLMError("empty reply, finish_reason=length: raise max_tokens or set LLM_REASONING_EFFORT=low")
        return LLMResponse(text, Usage(1, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), ms))


class CachedLLM:
    """On-disk cache around any LLMClient. Same (model_tag, system, user, max_tokens) -> same reply,
    so reruns are free, instant, and reproducible (important on rate-limited free tiers).
    A cache hit returns the originally recorded token usage, so cost numbers stay comparable;
    `hits` / `misses` tell you how many real calls were made. Only successful replies are stored."""

    def __init__(self, inner: LLMClient, path: str = ".cache/r3_llm_cache.jsonl", model_tag: str = ""):
        self.inner, self.path, self.model_tag = inner, path, model_tag
        self.hits = self.misses = 0
        self._store = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._store[rec["key"]] = rec

    def _key(self, system: str, user: str, max_tokens: int) -> str:
        raw = json.dumps([self.model_tag, system, user, max_tokens], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def complete(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse:
        key = self._key(system, user, max_tokens)
        rec = self._store.get(key)
        if rec:
            self.hits += 1
            return LLMResponse(rec["text"], Usage(**rec["usage"]))
        resp = self.inner.complete(system, user, max_tokens)
        self.misses += 1
        rec = {"key": key, "text": resp.text, "usage": {
            "calls": resp.usage.calls, "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens, "latency_ms": resp.usage.latency_ms}}
        self._store[key] = rec
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return resp
