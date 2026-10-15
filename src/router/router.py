"""Difficulty router: send each question to the cheapest strategy that is likely to be enough.

    DIRECT  (S0-like)  : no retrieval          -- off by default, see `allow_direct`
    SINGLE  (S2-like)  : one tuned-RAG pass
    AGENTIC (S4)       : planner + workers + verifier

One continuous `difficulty` score drives the decision, `route = AGENTIC if difficulty >= tau`.
Keeping a single score + single threshold is what makes the router's cost-accuracy curve
(`curve.py`) a simple threshold sweep, and keeps "no model training" true.

Methods
  heuristic : features only, zero LLM cost
  llm       : one short LLM classification call per question
  hybrid    : heuristic first; the LLM is called only when the heuristic score lands in the
              uncertain `band`, so most questions pay nothing
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .features import PaperIndex, difficulty_from_features, extract_features
from .llm import LLMClient
from .schemas import Route, RouteDecision, Usage

# Adaptive-RAG style labels -> difficulty
LABEL_SCORE = {"A": 0.0, "B": 0.2, "C": 0.9}

ROUTER_SYSTEM = """You route questions about recent AI research papers (agents, retrieval, reasoning) \
to the cheapest strategy that can answer them.
A = answerable from general knowledge, no lookup needed.
B = answerable from one passage of one paper (a single lookup).
C = needs evidence from several papers or sections: a comparison, an aggregation, or steps where \
one answer feeds the next question.
Questions about the specifics of a named paper are never A.
Reply with JSON only: {"label": "A" | "B" | "C", "reason": "<=12 words"}"""


def parse_label(text: str) -> Optional[str]:
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            lab = str(json.loads(m.group(0)).get("label", "")).strip().upper()
            if lab in LABEL_SCORE:
                return lab
        except (json.JSONDecodeError, AttributeError):
            pass
    m = re.search(r"\b([ABC])\b", text.upper())
    return m.group(1) if m else None


class Router:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        index: Optional[PaperIndex] = None,
        method: str = "hybrid",
        tau: float = 0.5,
        band: Tuple[float, float] = (0.3, 0.7),
        allow_direct: bool = False,
        probe: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    ):
        if method not in {"heuristic", "llm", "hybrid"}:
            raise ValueError("method must be heuristic | llm | hybrid")
        if method != "heuristic" and llm is None:
            raise ValueError(f"method={method!r} needs an llm client")
        self.llm, self.index, self.method = llm, index, method
        self.tau, self.band, self.allow_direct, self.probe = tau, band, allow_direct, probe

    def score(self, question: str) -> RouteDecision:
        """Compute the difficulty score and the LLM overhead. The threshold is applied in
        `route()`, so `curve.py` can reuse the cached score for every tau."""
        t0 = time.perf_counter()
        feats = extract_features(question, self.index, self.probe)
        h = difficulty_from_features(feats)
        usage, why = Usage(), []
        score, method, label = h, "heuristic", None

        why.append(f"{feats['n_papers']} paper(s) named" + (", comparison wording" if feats["compare"] else "")
                   + (", bridge wording" if feats["bridge"] else "") + f"; heuristic={h:.2f}")

        use_llm = self.method == "llm" or (self.method == "hybrid" and self.band[0] <= h <= self.band[1])
        if use_llm:
            resp = self.llm.complete(ROUTER_SYSTEM, question, max_tokens=60)
            usage.add(resp.usage)
            label = parse_label(resp.text)
            if label is None:
                why.append("LLM label unparseable; kept heuristic score")
            else:
                ls = LABEL_SCORE[label]
                if label == "A" and not self.allow_direct:
                    ls = LABEL_SCORE["B"]
                score = ls if self.method == "llm" else 0.5 * h + 0.5 * ls
                method = "llm" if self.method == "llm" else "hybrid-llm"
                why.append(f"LLM label {label}")
        elif self.method == "hybrid":
            method = "hybrid-heuristic"

        feats["llm_label"] = label
        return RouteDecision(Route.SINGLE, score, 0.0, method, feats, why, usage,
                             (time.perf_counter() - t0) * 1000)

    def route(self, question: str) -> RouteDecision:
        d = self.score(question)
        if self.allow_direct and d.features.get("llm_label") == "A" and d.features["n_papers"] == 0:
            d.route = Route.DIRECT
        else:
            d.route = Route.AGENTIC if d.difficulty >= self.tau else Route.SINGLE
        d.confidence = min(1.0, 2 * abs(d.difficulty - self.tau))
        return d
