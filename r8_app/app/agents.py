"""Agent interface, the demo agent, the setup catalogue and the registry.

Teammates plug in real systems by implementing `Agent` and registering it in
`REGISTRY` under its setup id ("S1" ... "S5"). Nothing else in the app changes:
the web app lists every setup and enables the ones that are registered.
"""
import asyncio
import os
import time
import zlib
from collections.abc import AsyncIterator
from typing import Protocol

from app.pipelines import build_team_agents
from app.rag import build_rag_agents
from app.schemas import AnswerObject, Citation, Cost, StreamEvent, TraceStep


class Agent(Protocol):
    name: str

    def stream(self, question: str) -> AsyncIterator[StreamEvent]:
        """Yield `step` events as work happens, then exactly one `final` event."""
        ...


SETUPS = [
    {"id": "S0", "name": "Closed book", "description": "No retrieval; the model answers from memory"},
    {"id": "S1", "name": "Vanilla RAG", "description": "Dense top-5 chunks, one pass"},
    {"id": "S2", "name": "Tuned RAG", "description": "Hybrid search, reranker, context headers"},
    {"id": "S3", "name": "Single agent", "description": "Search, read and reflect loop, capped at 8 steps"},
    {"id": "S4", "name": "Multi-agent", "description": "Planner, parallel workers and verifier"},
    {"id": "S5", "name": "Index-free agent", "description": "Keyword and grep tools only, no embeddings"},
]

_BRIEF = "Project brief"


def _c(i: int, title: str, passage: str, url: str) -> Citation:
    return Citation(id=i, paper_title=title, section=_BRIEF, passage=passage, url=url)


_ARXIV = "https://arxiv.org/abs/"

# Demo knowledge: short answers built only from what the project brief says about each paper.
_TOPICS = [
    {
        "keys": ["cost", "cheap", "router", "routing", "adaptive", "save", "expensive", "token"],
        "query": "adaptive routing retrieval cost",
        "subqs": ["when is extra retrieval worth its cost?", "how can a router skip it?"],
        "answer": (
            "A router can send easy questions down a cheap single-pass path and keep the full agent loop "
            "for hard ones [1]. RAGRouter-Bench provides a benchmark for judging those routing decisions [2]."
        ),
        "cites": [
            _c(1, "Adaptive-RAG", "Route each question to the cheapest strategy that can answer it.", _ARXIV + "2403.14403"),
            _c(2, "RAGRouter-Bench", "Benchmark for RAG routing decisions.", _ARXIV + "2602.00296"),
        ],
    },
    {
        "keys": ["inject", "attack", "hijack", "safe", "security", "malicious", "poison", "planted"],
        "query": "indirect prompt injection retrieval",
        "subqs": ["can retrieved text carry instructions?", "what defenses exist?"],
        "answer": (
            "Indirect prompt injection can survive retrieval, so an instruction planted in a document can "
            "reach the agent [1]. Grading retrieved evidence before using it, as CRAG does, is one defense [2]."
        ),
        "cites": [
            _c(1, "Overcoming the Retrieval Barrier", "Indirect prompt injection that survives retrieval.", _ARXIV + "2601.07072"),
            _c(2, "CRAG: Corrective RAG", "Grade retrieved evidence before using it.", _ARXIV + "2401.15884"),
        ],
    },
    {
        "keys": ["chunk", "header", "bm25", "context", "contextual", "index", "embedding", "dense", "hybrid", "rerank"],
        "query": "contextual retrieval chunk headers BM25",
        "subqs": ["how should chunks be prepared?", "does hybrid search help?"],
        "answer": (
            "Prefixing each chunk with its title and section header, then combining dense search with BM25, "
            "cut failed retrievals [1]. Retrievers should be judged inside the agent loop, not in isolation [2]."
        ),
        "cites": [
            _c(1, "Introducing Contextual Retrieval", "Context headers plus BM25 cut failed retrievals.", "https://www.anthropic.com/news/contextual-retrieval"),
            _c(2, "Rethinking Reasoning-Intensive Retrieval", "Judge retrievers inside the agent loop.", _ARXIV + "2605.04018"),
        ],
    },
    {
        "keys": ["multi-agent", "multiagent", "orchestrator", "worker", "parallel", "planner", "production", "context window"],
        "query": "orchestrator workers research system",
        "subqs": ["how is work split across agents?", "how is context kept small?"],
        "answer": (
            "Production research systems use an orchestrator that hands sub-questions to parallel workers [1]. "
            "Context has to be kept small as tool results pile up [2]."
        ),
        "cites": [
            _c(1, "How we built our multi-agent research system", "Orchestrator and workers in production.", "https://www.anthropic.com/engineering/multi-agent-research-system"),
            _c(2, "Effective context engineering for AI agents", "Keep context small as tool results pile up.", "https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents"),
        ],
    },
]

_DEFAULT_TOPIC = {
    "query": "agentic RAG versus single-shot RAG",
    "subqs": ["where does single-shot RAG break?", "what does the agent loop add?"],
    "answer": (
        "Standard RAG retrieves once and answers once, which breaks when evidence is spread across several "
        "papers [1]. Agentic systems loop through plan, search, read and verify, trading extra tokens and "
        "latency for better answers [2]."
    ),
    "cites": [
        _c(1, "A Survey on Agentic RAG", "Map of agent patterns: reflection, planning, tools.", _ARXIV + "2501.09136"),
        _c(2, "Survey of Deep Search Agents", "How deep search agents are built and evaluated.", _ARXIV + "2508.05668"),
    ],
}


def _pick_topic(question: str) -> dict:
    q = question.lower()
    scored = [(sum(k in q for k in t["keys"]), t) for t in _TOPICS]
    best_score, best = max(scored, key=lambda p: p[0])
    return best if best_score > 0 else _DEFAULT_TOPIC


class DemoAgent:
    """Scripted agent so the whole app can be built and demoed before real agents exist.

    Answers come from a handful of canned topics taken from the project brief; token and cost
    numbers are estimates. Real agents replace it by registering under S1 to S5.
    """

    name = "mock"

    async def stream(self, question: str) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        topic = _pick_topic(question)
        cites: list[Citation] = topic["cites"]
        script = [
            ("plan", "Split into sub-questions: " + "; ".join(topic["subqs"])),
            ("search", f"Hybrid search: '{topic['query']}'"),
            *[("read", f"Reading {c.paper_title}") for c in cites],
            ("verify", "Checking each claim against its cited passage"),
            ("answer", "Writing the final answer"),
        ]
        steps: list[TraceStep] = []
        for kind, message in script:
            delay_ms = 350 + zlib.crc32(message.encode()) % 450
            await asyncio.sleep(delay_ms / 1000)
            step = TraceStep(kind=kind, message=message, latency_ms=delay_ms)
            steps.append(step)
            yield StreamEvent(type="step", step=step)

        in_tok = 900 + 11 * len(question) + 380 * len(steps)
        out_tok = 60 + len(topic["answer"]) // 4
        answer = AnswerObject(
            question=question,
            answer=topic["answer"],
            citations=cites,
            trace=steps,
            cost=Cost(
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_s=round(time.perf_counter() - start, 2),
                usd=round(in_tok * 1.5e-7 + out_tok * 6e-7, 6),
            ),
            setup=self.name,
        )
        yield StreamEvent(type="final", answer=answer)


# Register real systems here, e.g. REGISTRY["S4"] = MultiAgent()
REGISTRY: dict[str, Agent] = {"mock": DemoAgent()}
REGISTRY.update(build_rag_agents())  # S1 and S2, when RAG_REPO_PATH points at the R1/R2 repo
REGISTRY.update(build_team_agents())  # S3, S4, S5, for whichever of R3-R6 are in that repo


def default_setup() -> str:
    return os.getenv("DEFAULT_AGENT", "mock")


def get_agent(setup: str | None = None) -> Agent:
    key = setup or default_setup()
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(f"Setup '{key}' is not available yet. Available: {sorted(REGISTRY)}") from None


def list_setups() -> list[dict]:
    demo = {"id": "mock", "name": "Demo agent", "description": "Scripted answers from the project brief, no live retrieval"}
    return [{**s, "available": s["id"] in REGISTRY} for s in [demo, *SETUPS]]
