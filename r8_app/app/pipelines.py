"""S3, S4 and S5 as runnable agents, built on the teammates' code without changing it.

What each teammate provides (all imported read-only from the R1-R6 repo):
  R2  src/retriever.py       PaperRetriever.search(query, top_k, mode)
  R3  src/router/*           Router, Planner, route_and_plan, trivial_plan
  R4  src/workers/*          RetrievalWorker (search + reflect loop), WorkerOrchestrator (waves)
  R5  src/mcp_tools/paper_tools.py   search / read / keyword_grep (no index needed)
  R6  src/verifier.py, src/injection_filter.py

None of them writes the final answer, so this module does that step (same prompt and model as S1/S2)
and maps everything into the shared answer object.

  S3  one RetrievalWorker loop (up to 8 steps) -> answer
  S4  Router -> (Planner -> parallel workers) -> injection filter -> answer -> claim verifier (+1 repair)
  S5  an LLM picks R5's keyword tools in a loop (no embeddings, no indices) -> answer

All LLM calls use the LLM_* environment variables. Heavy libraries load on first use.
"""
import asyncio
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from app.errors import AgentError
from app.rag import (
    SYSTEM_PROMPT,
    _find_repo,
    build_prompt,
    cited_ids,
    normalize_citations,
    to_citation,
)
from app.schemas import AnswerObject, Cost, StreamEvent, TraceStep

Emit = Callable[[TraceStep], None]

MAX_S3_STEPS = 8  # the brief's cap for the single agent
MAX_WORKER_STEPS = 3  # R3's stop_rules default for S4 workers
MAX_S5_STEPS = 6
MAX_PASSAGES = 8  # passages handed to the answer step
_CITE_NUM = re.compile(r"\[(\d+)\]")
_TRAILING_CITES = re.compile(r"([.!?])[ \t]*((?:\[\[[^\[\]]+\]\])+)")


# ---------- plumbing ----------

class Meter:
    """Wraps the teammates' LLM client and adds up tokens for one question (thread-safe)."""

    def __init__(self, inner: Any, usage_cls: Any) -> None:
        self._inner = inner
        self.usage = usage_cls()
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, max_tokens: int = 512):
        resp = self._inner.complete(system, user, max_tokens)
        with self._lock:
            self.usage.add(resp.usage)
        return resp


class TracingRetriever:
    """Passes calls to R2's retriever and reports every search as a live step."""

    def __init__(self, inner: Any, emit: Emit) -> None:
        self._inner = inner
        self._emit = emit

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid", **kw):
        t = time.perf_counter()
        hits = self._inner.search(query=query, top_k=top_k, mode=mode, **kw)
        self._emit(
            TraceStep(
                kind="search",
                message=f"{mode} search: “{query[:90]}” → {len(hits)} passages",
                latency_ms=int((time.perf_counter() - t) * 1000),
            )
        )
        return hits

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _load_by_path(path: Path):
    spec = importlib.util.spec_from_file_location(f"team_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TeamToolkit:
    """Loads the teammates' code once, on first use. `retriever` and `llm` can be injected (tests)."""

    def __init__(self, repo: Path, retriever: Any = None, llm: Any = None) -> None:
        self.repo = repo
        self._retriever = retriever
        self._llm = llm
        self._lock = threading.RLock()  # paper_index() calls mods() while holding it
        self._mods: dict[str, Any] | None = None
        self._index: Any = None

    # -- imports --
    def mods(self) -> dict[str, Any]:
        with self._lock:
            if self._mods is None:
                if str(self.repo) not in sys.path:
                    sys.path.insert(0, str(self.repo))
                try:
                    from src import router
                    from src.router import llm as router_llm
                    from src.workers.orchestrator import WorkerOrchestrator
                    from src.workers.worker import RetrievalWorker
                except ImportError as exc:
                    raise AgentError(
                        f"Could not load the teammates' code ({exc.name or exc}). "
                        "Install the libraries with: pip install -r requirements-rag.txt"
                    ) from exc
                self._mods = {
                    "router": router,
                    "router_llm": router_llm,
                    "Orchestrator": WorkerOrchestrator,
                    "Worker": RetrievalWorker,
                    "ClaimVerifier": _load_by_path(self.repo / "src" / "verifier.py").ClaimVerifier,
                    "InjectionFilter": self.injection_filter_cls(),
                }
            return self._mods

    def injection_filter_cls(self) -> Any:
        """R6's filter is plain Python, so it is loaded by path (S5 then needs no heavy libraries)."""
        return _load_by_path(self.repo / "src" / "injection_filter.py").InjectionFilter

    def retriever(self) -> Any:
        with self._lock:
            if self._retriever is None:
                try:
                    from src.retriever import PaperRetriever

                    self._retriever = PaperRetriever(
                        dense_db_path=str(self.repo / "indices" / "lancedb"),
                        bm25_index_dir=str(self.repo / "indices" / "bm25"),
                    )
                except Exception as exc:
                    raise AgentError(f"The retriever failed to start: {type(exc).__name__}: {exc}") from exc
            return self._retriever

    def paper_index(self) -> Any:
        with self._lock:
            if self._index is None:
                self._index = self.mods()["router"].PaperIndex.from_csv(str(self.repo / "papers.csv"))
            return self._index

    def meter(self) -> Meter:
        mods = self.mods()
        inner = self._llm
        if inner is None:
            key, model = os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
            if not (key and model):
                raise AgentError(
                    "This setup needs an LLM. Set LLM_API_KEY, LLM_MODEL and LLM_BASE_URL (see .env.example)."
                )
            base = os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            inner = mods["router_llm"].OpenAICompatibleClient(base_url=base, api_key=key, model=model)
        return Meter(inner, mods["router"].Usage)


def _scrub(exc: BaseException) -> str:
    key = os.getenv("LLM_API_KEY") or ""
    msg = f"{type(exc).__name__}: {exc}"
    return msg.replace(key, "***") if key else msg


async def _stream_blocking(work: Callable[[Emit], AnswerObject]) -> AsyncIterator[StreamEvent]:
    """Run blocking pipeline code in a thread and forward its steps live."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(step: TraceStep) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("step", step))

    def runner() -> None:
        try:
            result = work(emit)
            loop.call_soon_threadsafe(queue.put_nowait, ("final", result))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    threading.Thread(target=runner, daemon=True).start()
    while True:
        kind, value = await queue.get()
        if kind == "step":
            yield StreamEvent(type="step", step=value)
        elif kind == "final":
            yield StreamEvent(type="final", answer=value)
            return
        else:
            raise value if isinstance(value, AgentError) else AgentError(_scrub(value))


class _Steps:
    """Records steps for the answer object and forwards them live."""

    def __init__(self, emit: Emit) -> None:
        self.items: list[TraceStep] = []
        self._emit = emit
        self._lock = threading.Lock()

    def add(self, step: TraceStep) -> None:
        with self._lock:
            self.items.append(step)
        self._emit(step)

    def note(self, kind: str, message: str, t0: float) -> None:
        self.add(TraceStep(kind=kind, message=message, latency_ms=int((time.perf_counter() - t0) * 1000)))


def _passage(chunk: dict[str, Any]) -> dict[str, Any]:
    return {k: chunk.get(k, "") for k in ("paper_id", "chunk_id", "title", "section", "header", "text")}


def _merge_passages(chunks: list[dict[str, Any]], limit: int = MAX_PASSAGES) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in chunks:
        cid = str(c.get("chunk_id") or c.get("text", "")[:60])
        if cid in seen or not str(c.get("text", "")).strip():
            continue
        seen.add(cid)
        out.append(_passage(c))
        if len(out) >= limit:
            break
    return out


def _synthesize(meter: Meter, question: str, passages: list[dict[str, Any]], max_tokens: int = 700) -> str:
    if not passages:
        return "The corpus returned no passages for this question, so there is nothing to cite."
    resp = meter.complete(SYSTEM_PROMPT, build_prompt(question, passages), max_tokens)
    return normalize_citations(resp.text.strip())


def _filter_injections(toolkit: TeamToolkit, passages: list[dict[str, Any]], steps: _Steps) -> list[dict[str, Any]]:
    """R6's injection filter: flagged passages never reach the answer step."""
    t = time.perf_counter()
    report = toolkit.injection_filter_cls()().inspect_chunks(passages)
    flagged = {r["index"] for r in report["results"] if r.get("flagged")}
    if flagged:
        steps.note("verify", f"Injection filter dropped {len(flagged)} suspicious passage(s)", t)
    return [p for i, p in enumerate(passages) if i not in flagged]


def _finish(
    setup: str,
    question: str,
    text: str,
    passages: list[dict[str, Any]],
    steps: _Steps,
    meter: Meter,
    start: float,
) -> AnswerObject:
    u = meter.usage
    pin = float(os.getenv("LLM_PRICE_IN_PER_M") or 0)
    pout = float(os.getenv("LLM_PRICE_OUT_PER_M") or 0)
    return AnswerObject(
        question=question,
        answer=text,
        citations=[to_citation(i, passages[i - 1]) for i in cited_ids(text, len(passages))],
        trace=steps.items,
        cost=Cost(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            latency_s=round(time.perf_counter() - start, 2),
            usd=round((u.input_tokens * pin + u.output_tokens * pout) / 1e6, 6),
        ),
        setup=setup,
    )


# ---------- S3: single agent ----------

class SingleAgent:
    name = "S3"

    def __init__(self, toolkit: TeamToolkit) -> None:
        self._tk = toolkit

    async def stream(self, question: str) -> AsyncIterator[StreamEvent]:
        async for ev in _stream_blocking(lambda emit: self._run(question, emit)):
            yield ev

    def _run(self, question: str, emit: Emit) -> AnswerObject:
        tk, start = self._tk, time.perf_counter()
        mods, meter, steps = tk.mods(), tk.meter(), _Steps(emit)
        retriever = TracingRetriever(tk.retriever(), steps.add)
        worker = mods["Worker"](
            retriever, top_k=5, context_budget=3000, search_mode="hybrid", llm=meter, max_steps=MAX_S3_STEPS
        )
        sub = mods["router"].SubQuestion("q1", question)
        result = worker.run(sub, question)
        passages = _filter_injections(tk, _merge_passages(result["chunks"]), steps)
        t = time.perf_counter()
        text = _synthesize(meter, question, passages)
        steps.note("answer", f"Wrote the answer from {len(passages)} passages after {result['steps']} search step(s)", t)
        return _finish("S3", question, text, passages, steps, meter, start)


# ---------- S4: router + planner + workers + verifier ----------

_SUPPORT_SYSTEM = (
    "You check one claim against evidence. Reply with exactly one word: "
    "supported (the evidence states it), unsupported (the evidence contradicts it or does not mention it), "
    "or uncertain."
)
_REPAIR_SYSTEM = (
    "Rewrite the answer so every claim is supported by the numbered passages. Delete or fix the listed "
    "problem claims. Keep [n] citations, at most 4 sentences, no new facts."
)


class MultiAgent:
    name = "S4"

    def __init__(self, toolkit: TeamToolkit) -> None:
        self._tk = toolkit

    async def stream(self, question: str) -> AsyncIterator[StreamEvent]:
        async for ev in _stream_blocking(lambda emit: self._run(question, emit)):
            yield ev

    def _run(self, question: str, emit: Emit) -> AnswerObject:
        tk, start = self._tk, time.perf_counter()
        mods, meter, steps = tk.mods(), tk.meter(), _Steps(emit)
        rt = mods["router"]
        retriever = TracingRetriever(tk.retriever(), steps.add)

        # 1. R3: how much agent does this question need?
        t = time.perf_counter()
        router = rt.Router(llm=meter, index=tk.paper_index(), method="hybrid", tau=0.5)
        planner = rt.Planner(meter, tk.paper_index(), max_subquestions=4)
        decision, plan = rt.route_and_plan(question, router, planner)
        plan = plan or rt.trivial_plan(question, tk.paper_index())
        agentic = decision.route is rt.Route.AGENTIC
        steps.note(
            "plan",
            f"Router chose {decision.route.value} (difficulty {decision.difficulty:.2f})"
            + (": " + "; ".join(s.text for s in plan.sub_questions) if agentic else ""),
            t,
        )

        # 2. R4: workers run the plan wave by wave. Easy questions get one tuned search, no reflection.
        worker = mods["Worker"](
            retriever,
            top_k=5,
            context_budget=2000,
            search_mode="hybrid" if agentic else "reranked",
            llm=meter if agentic else None,
            max_steps=MAX_WORKER_STEPS,
        )
        outcome = mods["Orchestrator"](worker, max_parallel_workers=4).run(plan)
        failed = outcome["failed_workers"]
        if failed:
            steps.note("read", f"{len(failed)} worker(s) failed: {', '.join(failed)}", time.perf_counter())
        chunks = [c for r in outcome["results"].values() for c in r.get("chunks", [])]
        if not chunks and failed:
            first_error = next(r["error"] for r in outcome["results"].values() if "error" in r)
            raise AgentError(f"All workers failed: {first_error}")

        # 3. R6: drop suspicious passages, answer, then check every claim.
        passages = _filter_injections(tk, _merge_passages(chunks), steps)
        t = time.perf_counter()
        text = _synthesize(meter, question, passages)
        steps.note("answer", f"Wrote the answer from {len(passages)} passages", t)
        if passages:
            text = self._verify(tk, meter, steps, question, text, passages)
        return _finish("S4", question, text, passages, steps, meter, start)

    @staticmethod
    def _to_chunk_ids(text: str, passages: list[dict[str, Any]]) -> str:
        """R6 wants [[chunk_id]]; the app uses [n]."""

        def swap(m: re.Match) -> str:
            i = int(m.group(1))
            return f"[[{passages[i - 1]['chunk_id']}]]" if 1 <= i <= len(passages) else m.group(0)

        converted = _CITE_NUM.sub(swap, text)
        # R6 splits sentences at ". " so "claim. [[id]]" would orphan the citation: keep it before the stop.
        return _TRAILING_CITES.sub(lambda m: m.group(2) + m.group(1), converted)

    def _verify(self, tk, meter, steps, question, text, passages) -> str:
        verifier = tk.mods()["ClaimVerifier"](support_checker=lambda claim, cited: self._check(meter, claim, cited))
        for attempt in (0, 1):  # R3's stop rule: at most 1 repair round
            t = time.perf_counter()
            report = verifier.verify(self._to_chunk_ids(text, passages), passages)
            bad = [c for c in report["claims"] if c["citation_status"] != "valid" or c["support_status"] == "unsupported"]
            steps.note(
                "verify",
                f"Checked {report['claim_count']} claim(s): {report['claim_count'] - len(bad)} ok, {len(bad)} problem(s)",
                t,
            )
            if not bad or attempt == 1:
                break
            t = time.perf_counter()
            problems = "\n".join(f"- {c['claim']}" for c in bad)
            repaired = meter.complete(
                _REPAIR_SYSTEM,
                build_prompt(question, passages) + f"\n\nDraft answer:\n{text}\n\nProblem claims:\n{problems}\n\nRewrite:",
                700,
            )
            text = normalize_citations(repaired.text.strip()) or text
            steps.note("answer", "Rewrote the answer without the unsupported claims", t)
        return text

    @staticmethod
    def _check(meter: Meter, claim: str, cited: list[dict[str, Any]]) -> str:
        evidence = "\n\n".join(str(c.get("text", ""))[:1200] for c in cited)
        verdict = meter.complete(_SUPPORT_SYSTEM, f"Evidence:\n{evidence}\n\nClaim: {claim}", 5).text.strip().lower()
        for word in ("unsupported", "supported", "uncertain"):  # "unsupported" contains "supported"
            if verdict.startswith(word):
                return word
        return "uncertain"


# ---------- S5: index-free keyword agent ----------

_S5_SYSTEM = (
    "You research a question over a corpus of AI papers using only keyword tools. Reply with ONE JSON object:\n"
    '{"tool": "search", "query": "..."}            ranked keyword search over chunks\n'
    '{"tool": "keyword_grep", "keyword": "..."}    exact literal match (good for acronyms like CRAG)\n'
    '{"tool": "read", "paper_id": "...", "section": "optional"}   read part of one paper\n'
    '{"tool": "done"}                               when the evidence is enough\n'
    "No prose. Do not repeat a call."
)


def _load_paper_tools(repo: Path):
    return _load_by_path(repo / "src" / "mcp_tools" / "paper_tools.py")  # needs no torch or index


def _json_object(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class IndexFreeAgent:
    name = "S5"

    def __init__(self, toolkit: TeamToolkit, repo: Path) -> None:
        self._tk = toolkit
        self._repo = repo
        self._tools = None

    async def stream(self, question: str) -> AsyncIterator[StreamEvent]:
        async for ev in _stream_blocking(lambda emit: self._run(question, emit)):
            yield ev

    def _run(self, question: str, emit: Emit) -> AnswerObject:
        start = time.perf_counter()
        meter, steps = self._tk.meter(), _Steps(emit)
        try:
            tools = self._tools = self._tools or _load_paper_tools(self._repo)
        except Exception as exc:
            raise AgentError(f"Could not load the keyword tools: {type(exc).__name__}: {exc}") from exc

        evidence: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_calls: set[str] = set()

        def add_hits(hits: list[dict[str, Any]]) -> int:
            added = 0
            for h in hits:
                cid = str(h.get("chunk_id"))
                if cid in seen_ids or len(evidence) >= 12:
                    continue
                seen_ids.add(cid)
                evidence.append({**_passage({**h, "text": h.get("snippet", "")})})
                added += 1
            return added

        for step_no in range(1, MAX_S5_STEPS + 1):
            t = time.perf_counter()
            shown = "\n".join(f"[{i}] {e['title']} | {e['section']}: {e['text'][:200]}" for i, e in enumerate(evidence, 1))
            reply = meter.complete(_S5_SYSTEM, f"Question: {question}\n\nEvidence so far:\n{shown or '(none)'}", 150)
            call = _json_object(reply.text) or {"tool": "done"}
            key = json.dumps(call, sort_keys=True)
            if call.get("tool") == "done" or key in seen_calls:
                break
            seen_calls.add(key)
            try:
                if call["tool"] == "search":
                    hits = tools.search(str(call["query"]), 6)
                    note = f"keyword search “{call['query']}”"
                elif call["tool"] == "keyword_grep":
                    hits = tools.keyword_grep(str(call["keyword"]), 6)
                    note = f"grep “{call['keyword']}”"
                elif call["tool"] == "read":
                    body = tools.read(str(call["paper_id"]), str(call.get("section", "")))
                    blocks = re.findall(r"\[(.*?) \| (.*?) \| chunk=(.*?)\]\n(.*?)(?=\n\n\[|\Z)", body, re.S)
                    hits = [
                        {"paper_id": call["paper_id"], "title": a, "section": b, "chunk_id": c, "snippet": d[:900]}
                        for a, b, c, d in blocks[:3]
                    ]
                    note = f"read paper {call['paper_id']}"
                else:
                    break
            except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
                steps.note("search", f"Tool call failed ({type(exc).__name__}: {exc})", t)
                continue
            steps.note("search", f"{note} → {add_hits(hits)} new passages", t)

        passages = _filter_injections(self._tk, evidence[:MAX_PASSAGES], steps)
        t = time.perf_counter()
        text = _synthesize(meter, question, passages)
        steps.note("answer", f"Wrote the answer from {len(passages)} keyword-found passages", t)
        return _finish("S5", question, text, passages, steps, meter, start)


# ---------- registration ----------

def _has(repo: Path, *parts: str) -> bool:
    return (repo.joinpath(*parts)).exists()


def build_team_agents() -> dict[str, Any]:
    """S3, S4 and S5, for whichever teammates' code is present next to the app. Otherwise none."""
    repo = _find_repo()
    if repo is None:
        return {}
    toolkit = TeamToolkit(repo)
    agents: dict[str, Any] = {}
    if _has(repo, "src", "workers", "worker.py"):
        agents["S3"] = SingleAgent(toolkit)
    if all(_has(repo, "src", *p) for p in (("router", "router.py"), ("workers", "orchestrator.py"), ("verifier.py",), ("injection_filter.py",))):
        agents["S4"] = MultiAgent(toolkit)
    if _has(repo, "src", "mcp_tools", "paper_tools.py") and _has(repo, "corpus_v0"):
        agents["S5"] = IndexFreeAgent(toolkit, repo)
    return agents
