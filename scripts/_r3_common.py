"""Shared helpers for the R3 scripts. Run the scripts from the repo root."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # Windows consoles can choke on non-ASCII
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.router import CachedLLM, MockLLM, OpenAICompatibleClient, PaperIndex  # noqa: E402


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--questions", default="eval/r3_probe_questions.json",
                    help="JSON list of {id, question, ...}")
    ap.add_argument("--question", action="append",
                    help="route/plan this single question instead of a file (repeatable)")
    ap.add_argument("--papers", default="papers.csv")
    ap.add_argument("--method", default="hybrid", choices=["heuristic", "llm", "hybrid"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default="reports/r3")
    ap.add_argument("--mock", action="store_true", help="fake LLM (no API calls); checks the script, NOT quality")
    ap.add_argument("--no-cache", action="store_true", help="disable the on-disk LLM cache")


def load_questions(path: str, inline=None) -> List[dict]:
    if inline:
        return [{"id": f"cli{i+1}", "question": q} for i, q in enumerate(inline)]
    qs = json.load(open(path, encoding="utf-8"))
    for i, q in enumerate(qs):
        q.setdefault("id", f"q{i}")
    return qs


def expected_route(q: dict):
    """'agentic' / 'single' from either our probe set or R2's single-hop/multi-hop 'type'."""
    if q.get("expected_route") in ("agentic", "single"):
        return q["expected_route"]
    t = str(q.get("type", "")).lower()
    if "multi" in t:
        return "agentic"
    if "single" in t:
        return "single"
    return None


def _mock_responder(index: PaperIndex):
    cue = re.compile(r"\b(compare|differ|versus|vs|both|across|which papers)\b", re.I)

    def respond(system: str, user: str) -> str:
        if system.startswith("You route"):
            return json.dumps({"label": "C" if cue.search(user) else "B", "reason": "mock"})
        papers = sorted(index.mentioned(user))
        subs = ([{"id": f"q{i+1}", "text": f"What does paper {p} say about: {user}", "depends_on": []}
                 for i, p in enumerate(papers)] if len(papers) >= 2
                else [{"id": "q1", "text": user, "depends_on": []}])
        return json.dumps({"sub_questions": subs})
    return respond


def build_llm(args, index: PaperIndex, need_llm: bool = True):
    """Returns (llm, description). Real client reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL."""
    if args.mock:
        return MockLLM(_mock_responder(index)), "MOCK (no real LLM; results say nothing about quality)"
    if not need_llm:
        return None, "none"
    llm = OpenAICompatibleClient()
    desc = f"{llm.model} @ {llm.base_url}"
    if not args.no_cache:
        llm = CachedLLM(llm, ".cache/r3_llm_cache.jsonl", model_tag=desc)
    return llm, desc


def short(s: str, n: int = 70) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def need_env_help() -> str:
    return ("Missing LLM settings. In PowerShell:\n"
            '  $env:LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"\n'
            '  $env:LLM_API_KEY  = "<key>"\n'
            '  $env:LLM_MODEL    = "<model id>"\n'
            "or use --mock to run without an LLM (dry run only), or --method heuristic for routing only.")
