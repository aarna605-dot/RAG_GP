"""R3 step 2: SUB-QUESTION PLANNING. Routes each question, then decomposes the AGENTIC ones into
sub-questions with dependencies and parallel "waves" for R4's workers. SINGLE questions get a
one-sub-question plan at zero LLM cost.

    python scripts/r3_plan.py --question "Compare how CRAG and TrustRAG defend against noisy retrieval."
    python scripts/r3_plan.py                          # the probe questions
    python scripts/r3_plan.py --questions eval/all_questions.json --max-sub 4

Writes <out>/plans.json: a list of
    {id, question, route, difficulty, sub_questions:[{id,text,depends_on,paper_hints}], waves:[[ids]],
     fallback, notes, cost:{router_tokens, planner_tokens, total_tokens}}
R4 can load this file directly as fixtures, or call src.router.route_and_plan() itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _r3_common import add_common_args, build_llm, load_questions, need_env_help, short

from src.router import PaperIndex, Planner, Router, route_and_plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--max-sub", type=int, default=4, help="max sub-questions per plan")
    args = ap.parse_args()

    questions = load_questions(args.questions, args.question)
    index = PaperIndex.from_csv(args.papers)
    try:
        llm, desc = build_llm(args, index, need_llm=True)
    except KeyError as e:
        sys.exit(f"{need_env_help()}\n(missing {e})")
    router = Router(llm=llm, index=index, method=args.method, tau=args.tau)
    planner = Planner(llm, index, max_subquestions=args.max_sub)

    print(f"router={args.method} tau={args.tau} max_sub={args.max_sub} llm={desc}\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, fallbacks = [], 0
    for q in questions:
        decision, plan = route_and_plan(q["question"], router, planner)
        rt = decision.usage.total_tokens
        pt = plan.usage.total_tokens if plan else 0
        rec = {
            "id": q["id"], "question": q["question"], "route": decision.route.value,
            "difficulty": round(decision.difficulty, 4),
            "sub_questions": [s for s in plan.to_trace()["sub_questions"]] if plan else [],
            "waves": plan.to_trace()["waves"] if plan else [],
            "fallback": plan.fallback if plan else False,
            "notes": plan.notes if plan else [],
            "cost": {"router_tokens": rt, "planner_tokens": pt, "total_tokens": rt + pt},
        }
        results.append(rec)
        fallbacks += rec["fallback"]

        print(f"[{q['id']}] {decision.route.value} (difficulty {decision.difficulty:.2f}) {short(q['question'], 90)}")
        if plan is None:
            print("      (no retrieval)")
        for w, wave in enumerate(plan.waves() if plan else [], 1):
            for s in wave:
                dep = f"  needs {','.join(s.depends_on)}" if s.depends_on else ""
                print(f"      wave {w} | {s.id}: {s.text}{dep}")
        if rec["fallback"]:
            print(f"      ! planner fell back to one sub-question: {'; '.join(rec['notes'][-2:])}")
        print()

    json.dump(results, open(out / "plans.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    agentic = [r for r in results if r["route"] == "agentic"]
    print(f"{len(results)} questions, {len(agentic)} planned by the LLM, {fallbacks} fallbacks")
    if agentic:
        print(f"mean sub-questions per agentic plan: {sum(len(r['sub_questions']) for r in agentic)/len(agentic):.1f}")
    print(f"tokens: {sum(r['cost']['total_tokens'] for r in results)} total")
    if hasattr(llm, "hits"):
        print(f"cache: {llm.hits} hits, {llm.misses} real API calls")
    print(f"wrote {out / 'plans.json'}")


if __name__ == "__main__":
    main()
