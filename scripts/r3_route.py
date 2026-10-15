"""R3 step 1: DIFFICULTY ROUTING. Decides, for every question, SINGLE (cheap one-pass RAG) or AGENTIC.

    python scripts/r3_route.py                                   # the probe questions, hybrid router
    python scripts/r3_route.py --question "Compare CRAG and TrustRAG on noisy context."
    python scripts/r3_route.py --questions eval/all_questions.json --method hybrid --tau 0.5
    python scripts/r3_route.py --method heuristic                # no LLM, no API key needed

Writes (under --out, default reports/r3/):
    route_decisions.jsonl  one routing decision per question (what R4 / the harness reads)
    router_scores.json     {id: [difficulty, router_tokens]} (input for src.router.curve)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from _r3_common import add_common_args, build_llm, expected_route, load_questions, need_env_help, short

from src.router import PaperIndex, Router


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--allow-direct", action="store_true", help="enable the no-retrieval route (ablation)")
    args = ap.parse_args()

    questions = load_questions(args.questions, args.question)
    index = PaperIndex.from_csv(args.papers)
    try:
        llm, desc = build_llm(args, index, need_llm=args.method != "heuristic")
    except KeyError as e:
        sys.exit(f"{need_env_help()}\n(missing {e})")
    router = Router(llm=llm, index=index, method=args.method, tau=args.tau, allow_direct=args.allow_direct)

    print(f"router={args.method} tau={args.tau} llm={desc}\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, scores = [], {}
    for q in questions:
        d = router.route(q["question"])
        row = {
            "id": q["id"], "question": q["question"], "route": d.route.value,
            "difficulty": round(d.difficulty, 4), "confidence": round(d.confidence, 4),
            "method": d.method, "papers": d.features["papers"], "llm_label": d.features.get("llm_label"),
            "rationale": d.rationale, "router_tokens": d.usage.total_tokens,
            "router_usd": round(d.usage.usd, 6), "latency_ms": round(d.latency_ms, 1),
            "expected_route": expected_route(q),
        }
        rows.append(row)
        scores[q["id"]] = [d.difficulty, d.usage.total_tokens]
        mark = ""
        if row["expected_route"]:
            mark = "ok " if row["expected_route"] == row["route"] else "MISS"
        print(f"{q['id']:>5} {d.route.value:<8} {d.difficulty:.2f} {mark:<4} {short(q['question'])}")

    with open(out / "route_decisions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(scores, open(out / "router_scores.json", "w"), indent=1)

    n = len(rows)
    c = Counter(r["route"] for r in rows)
    calls = sum(1 for r in rows if r["router_tokens"] > 0)
    print(f"\n{n} questions -> " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())))
    print(f"router LLM used on {calls}/{n} questions, {sum(r['router_tokens'] for r in rows)} tokens total")
    if hasattr(llm, "hits"):
        print(f"cache: {llm.hits} hits, {llm.misses} real API calls")
    labelled = [r for r in rows if r["expected_route"]]
    if labelled:
        ok = sum(r["expected_route"] == r["route"] for r in labelled)
        print(f"agreement with expected_route labels: {ok}/{len(labelled)} (labels are judgments; read the MISS rows)")
    print(f"wrote {out / 'route_decisions.jsonl'} and {out / 'router_scores.json'}")


if __name__ == "__main__":
    main()
