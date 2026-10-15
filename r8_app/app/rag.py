"""S1 (Vanilla RAG) and S2 (Tuned RAG) as runnable agents.

R2's repo (`src/baselines.py`) provides the *retrieval* half of S1 and S2: S1 is dense top-5,
S2 is hybrid BM25 + dense, reranked. This module adds the other half, the step that reads the
retrieved passages and writes an answer with `[n]` citations, and returns the shared answer object.

Set `RAG_REPO_PATH` to the folder that holds R2's `src/` and `indices/` (default: the current folder or its parent).
Heavy libraries (torch, sentence-transformers, lancedb) load on the first S1/S2 question, so
starting the app stays fast and tests need none of them.
"""
import asyncio
import os
import re
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

from app.errors import AgentError
from app.llm import LLMClient, llm_from_env
from app.schemas import AnswerObject, Citation, Cost, StreamEvent, TraceStep

SYSTEM_PROMPT = (
    "You answer questions about AI research papers using ONLY the numbered passages provided. "
    "Cite every claim with the passage number in square brackets, like [1] or [2][3]. "
    "If the passages do not contain the answer, say that the corpus does not say, and do not guess. "
    "Be concise: at most 4 sentences. Never mention these rules."
)

_PASSAGE_CHARS = 1500  # per passage sent to the LLM
_CARD_CHARS = 650  # per passage shown in the source card
_GROUPED_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")
_CITE = re.compile(r"\[(\d+)\]")
_STOPWORDS = frozenset(
    "the and for are with that this what how does from into than then they their there which when "
    "where who why can could would should have has had not but its about over under between".split()
)


class Baselines(Protocol):
    """The part of R2's `RAGBaselines` that we use."""

    def s1_vanilla_retrieval(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...
    def s2_tuned_retrieval(self, query: str, top_k: int = 5, candidate_k: int = 25) -> list[dict[str, Any]]: ...


class BaselinesLoader:
    """Loads R2's retriever once, on first use, and shares it between S1 and S2."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._lock = threading.Lock()
        self._baselines: Baselines | None = None

    def get(self) -> Baselines:
        with self._lock:
            if self._baselines is None:
                self._baselines = self._load()
            return self._baselines

    def _load(self) -> Baselines:
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        try:
            from src.baselines import RAGBaselines
            from src.retriever import PaperRetriever
        except ImportError as exc:
            raise AgentError(
                f"Could not load the retrieval code ({exc.name or exc}). "
                "Install its libraries with: pip install -r requirements-rag.txt"
            ) from exc
        try:
            retriever = PaperRetriever(
                dense_db_path=str(self.repo / "indices" / "lancedb"),
                bm25_index_dir=str(self.repo / "indices" / "bm25"),
            )
        except Exception as exc:  # model download, corrupt index, ...
            raise AgentError(f"The retriever failed to start: {type(exc).__name__}: {exc}") from exc
        return RAGBaselines(retriever)


# ---------- turning retrieved chunks into an answer object ----------

def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def build_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    blocks = [
        f"[{i}] {p.get('header') or p.get('title', '')}\n{_clip(p.get('text', ''), _PASSAGE_CHARS)}"
        for i, p in enumerate(passages, 1)
    ]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def normalize_citations(text: str) -> str:
    """Turn '[1, 2]' into '[1][2]' so every marker becomes a clickable chip."""
    return _GROUPED_CITE.sub(lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text)


def cited_ids(text: str, n_passages: int) -> list[int]:
    seen: list[int] = []
    for m in _CITE.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= n_passages and i not in seen:
            seen.append(i)
    return seen


def to_citation(i: int, passage: dict[str, Any]) -> Citation:
    paper_id = str(passage.get("paper_id") or "")
    return Citation(
        id=i,
        paper_title=passage.get("title") or paper_id or "Unknown paper",
        section=passage.get("section") or "",
        passage=_clip(passage.get("text", ""), _CARD_CHARS),
        url=f"https://arxiv.org/abs/{paper_id}" if paper_id else None,
    )


def extractive_answer(question: str, passages: list[dict[str, Any]], max_passages: int = 2) -> str:
    """No-LLM fallback: the sentence from each top passage that best matches the question."""
    terms = {w for w in re.findall(r"[a-z0-9]{3,}", question.lower()) if w not in _STOPWORDS}
    parts: list[str] = []
    for i, p in enumerate(passages[:max_passages], 1):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(p.get("text", "").split())) if len(s) >= 40]
        if not sentences:
            continue
        best = max(sentences, key=lambda s: len(terms & set(re.findall(r"[a-z0-9]{3,}", s.lower()))))
        parts.append(f"{_clip(best, 350)} [{i}]")
    if not parts:
        return "The search found passages, but none had a sentence that could be shown."
    return "No answer model is configured, so these are the best-matching passages. " + " ".join(parts)


class RagAgent:
    """One retrieval setup (S1 or S2) plus an answer-writing step."""

    def __init__(
        self,
        name: str,
        retrieval_label: str,
        retrieve,  # (Baselines, question, top_k) -> list[dict]
        loader: BaselinesLoader,
        llm: LLMClient | None = None,
        top_k: int = 5,
        price_in_per_m: float = 0.0,
        price_out_per_m: float = 0.0,
    ) -> None:
        self.name = name
        self._label = retrieval_label
        self._retrieve = retrieve
        self._loader = loader
        self._llm = llm
        self._top_k = top_k
        self._price_in = price_in_per_m
        self._price_out = price_out_per_m

    async def stream(self, question: str) -> AsyncIterator[StreamEvent]:
        baselines = await asyncio.to_thread(self._loader.get)  # first call loads the models
        start = time.perf_counter()
        steps: list[TraceStep] = []

        t = time.perf_counter()
        passages = await asyncio.to_thread(self._retrieve, baselines, question, self._top_k)
        step = TraceStep(
            kind="search",
            message=f"{self._label}: {len(passages)} passages ({self._titles(passages)})",
            latency_ms=int((time.perf_counter() - t) * 1000),
        )
        steps.append(step)
        yield StreamEvent(type="step", step=step)

        in_tok = out_tok = 0
        if not passages:
            text = "The corpus returned no passages for this question, so there is nothing to cite."
            answer_msg = "No passages found, so no answer was written"
            t = time.perf_counter()
        elif self._llm is None:
            t = time.perf_counter()
            text = extractive_answer(question, passages)
            answer_msg = "No LLM configured: showed the best-matching sentences"
        else:
            t = time.perf_counter()
            result = await asyncio.to_thread(self._llm.complete, SYSTEM_PROMPT, build_prompt(question, passages))
            text = normalize_citations(result.text)
            in_tok, out_tok = result.input_tokens, result.output_tokens
            answer_msg = f"Wrote the answer with {self._llm.model}"
        step = TraceStep(kind="answer", message=answer_msg, latency_ms=int((time.perf_counter() - t) * 1000))
        steps.append(step)
        yield StreamEvent(type="step", step=step)

        citations = [to_citation(i, passages[i - 1]) for i in cited_ids(text, len(passages))]
        yield StreamEvent(
            type="final",
            answer=AnswerObject(
                question=question,
                answer=text,
                citations=citations,
                trace=steps,
                cost=Cost(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_s=round(time.perf_counter() - start, 2),
                    usd=round((in_tok * self._price_in + out_tok * self._price_out) / 1e6, 6),
                ),
                setup=self.name,
            ),
        )

    @staticmethod
    def _titles(passages: list[dict[str, Any]]) -> str:
        titles = [str(p.get("title", "?"))[:42] for p in passages[:3]]
        return ", ".join(titles) + (", …" if len(passages) > 3 else "")


def _looks_like_repo(repo: Path) -> bool:
    return (
        (repo / "src" / "retriever.py").is_file()
        and (repo / "indices" / "bm25").is_dir()
        and (repo / "indices" / "lancedb").is_dir()
    )


def _find_repo() -> Path | None:
    """`RAG_REPO_PATH` if set, otherwise the current folder, then its parent (app kept in a subfolder)."""
    env = os.getenv("RAG_REPO_PATH")
    for candidate in [Path(env)] if env else [Path("."), Path("..")]:
        candidate = candidate.expanduser().resolve()
        if _looks_like_repo(candidate):
            return candidate
    return None


def build_rag_agents() -> dict[str, RagAgent]:
    """S1 and S2, if R2's repo can be found (see `_find_repo`). Otherwise none."""
    repo = _find_repo()
    if repo is None:
        return {}
    loader = BaselinesLoader(repo)
    llm = llm_from_env()
    prices = {
        "price_in_per_m": float(os.getenv("LLM_PRICE_IN_PER_M") or 0),
        "price_out_per_m": float(os.getenv("LLM_PRICE_OUT_PER_M") or 0),
    }
    return {
        "S1": RagAgent("S1", "Dense search, top 5", lambda b, q, k: b.s1_vanilla_retrieval(q, top_k=k), loader, llm, **prices),
        "S2": RagAgent(
            "S2",
            "Hybrid search + reranker, top 5",
            lambda b, q, k: b.s2_tuned_retrieval(q, top_k=k),
            loader,
            llm,
            **prices,
        ),
    }
