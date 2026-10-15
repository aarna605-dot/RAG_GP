"""Cheap, zero-LLM-cost signals that predict whether a question needs the agent loop.

Idea (Adaptive-RAG, RAGRouter-Bench): most of the time the cheapest sufficient strategy can be
guessed from the question alone. In this corpus the strongest signal is how many *distinct papers*
the question names, then comparison / aggregation / bridge wording.
"""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Aliases too generic to identify a single paper.
GENERIC_ALIASES = {
    "RAG", "LLM", "LLMS", "AI", "QA", "RL", "MCP", "NLP", "GPT", "IR", "KG", "SOTA", "API",
}
_ARXIV_ID = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


class PaperIndex:
    """Finds which corpus papers a question mentions (title, short name, acronym, arXiv id)."""

    def __init__(self, papers: List[Dict[str, str]]):
        self.ids: Set[str] = {p["arxiv_id"] for p in papers}
        self._phrases: Dict[str, str] = {}              # normalized title / title-prefix -> id
        alias_owner: Dict[str, Set[str]] = defaultdict(set)

        for p in papers:
            pid, title = p["arxiv_id"], p["title"].strip()
            if len(_norm(title).split()) >= 2:
                self._phrases[_norm(title)] = pid
            prefix = title.split(":")[0].strip() if ":" in title else ""
            if prefix:
                if len(prefix.split()) == 1:
                    alias_owner[prefix].add(pid)       # e.g. TrustRAG, Adaptive-RAG
                elif len(_norm(prefix).split()) >= 2:
                    self._phrases[_norm(prefix)] = pid
            else:
                # initials alias (Corrective Retrieval Augmented Generation -> CRAG), short titles only,
                # otherwise long Title-Case titles produce junk acronyms
                words = re.findall(r"[A-Za-z]+", title)
                if 3 <= len(words) <= 5 and all(w[0].isupper() for w in words):
                    alias_owner["".join(w[0] for w in words).upper()].add(pid)

        self._aliases: Dict[str, str] = {}
        for alias, owners in alias_owner.items():
            if len(owners) == 1 and alias.upper() not in GENERIC_ALIASES and 3 <= len(alias) <= 20:
                self._aliases[alias] = next(iter(owners))
        self._alias_res = [
            (re.compile(r"(?<![A-Za-z0-9])" + re.escape(a).replace(r"\-", r"[-\s]?") + r"(?![A-Za-z0-9])"), pid)
            for a, pid in self._aliases.items()
        ]

    @classmethod
    def from_csv(cls, path: str = "papers.csv") -> "PaperIndex":
        with open(path, newline="", encoding="utf-8") as f:
            return cls(list(csv.DictReader(f)))

    def mentioned(self, question: str) -> Set[str]:
        found: Set[str] = set()
        q_norm = f" {_norm(question)} "
        for phrase, pid in self._phrases.items():
            if f" {phrase} " in q_norm:
                found.add(pid)
        for rx, pid in self._alias_res:
            if rx.search(question):
                found.add(pid)
        for m in _ARXIV_ID.findall(question):
            found.add(m)
        return found


_COMPARE = re.compile(
    r"\b(compare[sd]?|comparison|contrast|versus|vs\.?|differ(?:s|ence|ences|ently)?|"
    r"better than|trade-?offs?|relative to|similarit(?:y|ies)|advantages? over)\b", re.I)
_AGGREGATE = re.compile(
    r"\b(both|each|every|across|overall|how many|which (?:papers|methods|approaches|systems)|"
    r"common|shared|all (?:of )?the)\b", re.I)
_BRIDGE = re.compile(
    r"\b(the (?:paper|method|model|approach|system|authors?|benchmark) (?:that|which|who|proposed|introduced)|"
    r"whose|proposed by|introduced by|same (?:authors?|group|idea)|which of (?:the|these))\b", re.I)
_CONJ = re.compile(r"\b(and|while|whereas|then|also|as well as|but)\b", re.I)
_WH = re.compile(r"\b(what|how|why|which|who|when|where)\b", re.I)

# Hand-set priors (NOT trained). The threshold tau is what gets calibrated, on the cost-accuracy curve.
WEIGHTS = {
    "bias": -1.0,
    "two_papers": 1.2,
    "three_papers": 0.6,
    "compare": 1.0,
    "aggregate": 0.5,
    "bridge": 0.5,
    "multi_clause": 0.4,
    "long": 0.3,
    "multi_question": 0.5,
    "spread": 0.4,
}


def extract_features(
    question: str,
    index: Optional[PaperIndex] = None,
    probe: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """`probe` is an optional cheap retriever call, e.g. `PaperRetriever().search_bm25`
    (about 8 ms, no LLM). If the top hits scatter over many papers with no clear winner,
    the evidence is probably spread out."""
    q = question.strip()
    papers = index.mentioned(q) if index else set()
    feats: Dict[str, Any] = {
        "n_words": len(q.split()),
        "n_papers": len(papers),
        "papers": sorted(papers),
        "compare": bool(_COMPARE.search(q)),
        "aggregate": bool(_AGGREGATE.search(q)),
        "bridge": bool(_BRIDGE.search(q)),
        "n_conj": len(_CONJ.findall(q)),
        "n_wh": len(_WH.findall(q)),
        "n_questions": max(1, q.count("?")),
    }
    if probe is not None:
        hits = probe(q)[:10]
        counts: Dict[str, int] = defaultdict(int)
        for h in hits:
            counts[h["paper_id"]] += 1
        top_share = (max(counts.values()) / len(hits)) if hits else 1.0
        feats["probe_papers"] = len(counts)
        feats["probe_top_share"] = round(top_share, 2)
    return feats


def difficulty_from_features(f: Dict[str, Any], w: Dict[str, float] = WEIGHTS) -> float:
    z = w["bias"]
    z += w["two_papers"] * (f["n_papers"] >= 2)
    z += w["three_papers"] * (f["n_papers"] >= 3)
    z += w["compare"] * f["compare"]
    z += w["aggregate"] * f["aggregate"]
    z += w["bridge"] * f["bridge"]
    z += w["multi_clause"] * (f["n_conj"] >= 2 or f["n_wh"] >= 2)
    z += w["long"] * (f["n_words"] >= 25)
    z += w["multi_question"] * (f["n_questions"] >= 2)
    if "probe_papers" in f:
        z += w["spread"] * (f["probe_papers"] >= 5 and f["probe_top_share"] <= 0.4)
    return 1.0 / (1.0 + math.exp(-z))
