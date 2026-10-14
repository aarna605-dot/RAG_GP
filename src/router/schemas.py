"""Shared data types for R3 (router + planner).

Everything here is plain dataclasses with a `to_dict()` so it drops straight into
the answer object's `trace` (R8 streams it) and into JSON logs (R7 reads it).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Route(str, Enum):
    DIRECT = "direct"      # no retrieval            -> behaves like S0
    SINGLE = "single"      # one tuned-RAG pass      -> behaves like S2
    AGENTIC = "agentic"    # plan + workers + verify -> S4 pipeline


# Reference prices (USD per 1M tokens) used ONLY to turn tokens into a dollar figure.
# Replace with whatever R7 fixes for the final results table.
REFERENCE_PRICE_PER_MTOK = {"input": 0.15, "output": 0.60}


@dataclass
class Usage:
    """LLM usage of one component. Router/planner calls MUST be counted:
    routing overhead is part of the cost the router has to beat."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usd(self) -> float:
        p = REFERENCE_PRICE_PER_MTOK
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1e6

    def add(self, other: "Usage") -> "Usage":
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.latency_ms += other.latency_ms
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["usd"] = round(self.usd, 6)
        return d


@dataclass
class RouteDecision:
    route: Route
    difficulty: float                 # 0 (trivial) .. 1 (hard); the ONLY value the threshold acts on
    confidence: float                 # 0..1, distance from the threshold
    method: str                       # "heuristic" | "llm" | "hybrid-heuristic" | "hybrid-llm"
    features: Dict[str, Any] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    def to_trace(self) -> Dict[str, Any]:
        return {
            "type": "router",
            "route": self.route.value,
            "difficulty": round(self.difficulty, 3),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "rationale": self.rationale,
            "features": self.features,
            "cost": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class SubQuestion:
    id: str
    text: str                                         # may contain {qN} placeholders
    depends_on: List[str] = field(default_factory=list)
    paper_hints: List[str] = field(default_factory=list)   # arXiv ids detected in the text

    def resolve(self, answers: Dict[str, str]) -> str:
        """Fill {qN} placeholders with the answers of earlier sub-questions."""
        out = self.text
        for dep in self.depends_on:
            out = out.replace("{" + dep + "}", answers.get(dep, ""))
        return out


@dataclass
class Plan:
    question: str
    sub_questions: List[SubQuestion]
    fallback: bool = False                # True if decomposition failed -> single sub-question
    usage: Usage = field(default_factory=Usage)
    notes: List[str] = field(default_factory=list)

    def waves(self) -> List[List[SubQuestion]]:
        """Topological levels. Everything inside one wave can run in parallel (R4 workers)."""
        done: set = set()
        remaining = list(self.sub_questions)
        levels: List[List[SubQuestion]] = []
        while remaining:
            ready = [s for s in remaining if all(d in done for d in s.depends_on)]
            if not ready:  # defensive: validated plans never hit this
                ready = remaining
            levels.append(ready)
            done.update(s.id for s in ready)
            remaining = [s for s in remaining if s not in ready]
        return levels

    def to_trace(self) -> Dict[str, Any]:
        return {
            "type": "plan",
            "fallback": self.fallback,
            "sub_questions": [asdict(s) for s in self.sub_questions],
            "waves": [[s.id for s in w] for w in self.waves()],
            "notes": self.notes,
            "cost": self.usage.to_dict(),
        }
