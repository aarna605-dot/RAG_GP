"""Planner: split a hard question into sub-questions that R4's workers can run in parallel.

Output is a small DAG. Independent sub-questions share a wave and run in parallel; a sub-question
that needs an earlier answer references it as {qN} and lands in a later wave.
Safe by construction: any failure (bad JSON, cycle, empty plan) falls back to one sub-question
equal to the original question, i.e. the agent degrades to S3 behaviour instead of crashing.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .features import PaperIndex
from .llm import LLMClient
from .router import Router
from .schemas import Plan, Route, RouteDecision, SubQuestion, Usage

PLANNER_SYSTEM = """You split a research question about AI papers into the fewest sub-questions \
needed to answer it. Each sub-question will be answered by a separate search over a corpus of arXiv papers.
Rules:
- Write each sub-question as a complete natural-language question, not a keyword query. Name the paper \
or method it is about, using the names from the original question.
- Cover every part of the original question, including constraints such as "which is cheaper" or \
"how do they differ". Do not drop a part.
- For a comparison of named items, use one sub-question per item. Do NOT add a final "compare" step; \
the orchestrator does the synthesis.
- If the question asks to find a set of papers and then say something about each, use a first \
sub-question that finds them and a second that depends on it, referring to the first answer as {q1} \
and listing "q1" in depends_on.
- Use at most {max_n} sub-questions. If one search is enough, return exactly one.
Example: "Which papers study X, and how do their Y differ?" becomes
{"sub_questions": [{"id": "q1", "text": "Which papers study X?", "depends_on": []}, \
{"id": "q2", "text": "What is the Y in each of these papers: {q1}?", "depends_on": ["q1"]}]}
Reply with JSON only:
{"sub_questions": [{"id": "q1", "text": "...", "depends_on": []}]}"""


def _extract_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def _has_cycle(deps: Dict[str, List[str]]) -> bool:
    state: Dict[str, int] = {}

    def visit(n: str) -> bool:
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        if any(visit(d) for d in deps[n]):
            return True
        state[n] = 2
        return False

    return any(visit(n) for n in deps)


def validate(raw: dict, max_n: int) -> Tuple[Optional[List[SubQuestion]], List[str]]:
    """Return (sub_questions, notes); sub_questions is None if the plan is unusable."""
    notes: List[str] = []
    items = raw.get("sub_questions") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        return None, ["no sub_questions list"]
    subs: List[SubQuestion] = []
    seen_text = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not str(it.get("text", "")).strip():
            notes.append(f"dropped empty item {i}")
            continue
        sid = str(it.get("id") or f"q{i + 1}")
        text = str(it["text"]).strip()
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if sid in {s.id for s in subs} or key in seen_text:
            notes.append(f"dropped duplicate {sid}")
            continue
        seen_text.add(key)
        deps = [str(d) for d in (it.get("depends_on") or [])]
        subs.append(SubQuestion(sid, text, deps))
    if len(subs) > max_n:
        notes.append(f"truncated {len(subs)} -> {max_n} sub-questions")
        subs = subs[:max_n]
    ids = {s.id for s in subs}
    for s in subs:
        bad = [d for d in s.depends_on if d not in ids or d == s.id]
        if bad:
            notes.append(f"{s.id}: removed unknown/self deps {bad}")
            s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]
    if not subs:
        return None, notes + ["no valid sub-questions"]
    if _has_cycle({s.id: s.depends_on for s in subs}):
        return None, notes + ["dependency cycle"]
    return subs, notes


class Planner:
    def __init__(self, llm: LLMClient, index: Optional[PaperIndex] = None,
                 max_subquestions: int = 4, max_retries: int = 1):
        self.llm, self.index = llm, index
        self.max_n, self.max_retries = max_subquestions, max_retries

    def plan(self, question: str) -> Plan:
        usage, notes = Usage(), []
        system = PLANNER_SYSTEM.replace("{max_n}", str(self.max_n))
        user = question
        for attempt in range(self.max_retries + 1):
            resp = self.llm.complete(system, user, max_tokens=400)
            usage.add(resp.usage)
            raw = _extract_json(resp.text)
            subs, n = validate(raw, self.max_n) if raw else (None, ["unparseable JSON"])
            notes += n
            if subs:
                if self.index:
                    for s in subs:
                        s.paper_hints = sorted(self.index.mentioned(s.text))
                return Plan(question, subs, False, usage, notes)
            user = question + "\n\n(Previous reply was invalid: " + "; ".join(n) + ". Reply with valid JSON only.)"
        notes.append("fell back to a single sub-question")
        hints = sorted(self.index.mentioned(question)) if self.index else []
        return Plan(question, [SubQuestion("q1", question, [], hints)], True, usage, notes)


def trivial_plan(question: str, index: Optional[PaperIndex] = None) -> Plan:
    """Zero-cost plan for the SINGLE route: the question is its own only sub-question."""
    hints = sorted(index.mentioned(question)) if index else []
    return Plan(question, [SubQuestion("q1", question, [], hints)])


def route_and_plan(question: str, router: Router, planner: Planner) -> Tuple[RouteDecision, Optional[Plan]]:
    """The single entry point R4/R8 call. Only AGENTIC pays for a planner call."""
    decision = router.route(question)
    if decision.route is Route.AGENTIC:
        return decision, planner.plan(question)
    if decision.route is Route.SINGLE:
        return decision, trivial_plan(question, router.index)
    return decision, None