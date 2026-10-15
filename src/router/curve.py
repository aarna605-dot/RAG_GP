"""Router cost-accuracy curve (R3 deliverable) and threshold calibration.

Inputs (produced by R7's harness, one row per question per setup):
  results.jsonl : {"qid", "setup", "<metric>": float, "tokens": int, "usd": float, "latency_s": float}
  questions.json: [{"id", "question", ...}]

For each threshold tau, questions with difficulty >= tau go to the STRONG setup (S4), the rest to the
CHEAP setup (S2). Router overhead (its own LLM calls) is added to every question's cost, because the
router has to beat its own cost. Compared against always-S4 (router off).

Because 2 accuracy points on 200 questions is only 4 questions, tau is chosen by k-fold cross-validation
and the headline numbers come with bootstrap confidence intervals.

    python -m src.router.curve --questions eval/all_questions.json --results results/results.jsonl \
        --cheap S2 --strong S4 --metric judge --cost tokens --out reports/r3
    python -m src.router.curve --demo        # synthetic data, only to see the pipeline work
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Item:
    qid: str
    score: float          # router difficulty
    overhead: float       # router's own cost, same unit as costs
    cheap_acc: float
    cheap_cost: float
    strong_acc: float
    strong_cost: float


@dataclass
class Point:
    tau: float
    frac_agentic: float
    acc: float
    cost: float
    acc_delta: float      # vs always-strong, in accuracy units (0.02 = 2 points)
    cost_cut: float       # fraction saved vs always-strong


def evaluate(items: Sequence[Item], tau: float) -> Tuple[float, float, float]:
    acc = cost = n_strong = 0.0
    for it in items:
        use_strong = it.score >= tau
        acc += it.strong_acc if use_strong else it.cheap_acc
        cost += it.overhead + (it.strong_cost if use_strong else it.cheap_cost)
        n_strong += use_strong
    n = len(items)
    return acc / n, cost / n, n_strong / n


def baselines(items: Sequence[Item]) -> Dict[str, Tuple[float, float]]:
    n = len(items)
    oracle_acc = sum(max(i.cheap_acc, i.strong_acc) for i in items) / n
    oracle_cost = sum(i.cheap_cost if i.cheap_acc >= i.strong_acc else i.strong_cost for i in items) / n
    return {
        "always_cheap": (sum(i.cheap_acc for i in items) / n, sum(i.cheap_cost for i in items) / n),
        "always_strong": (sum(i.strong_acc for i in items) / n, sum(i.strong_cost for i in items) / n),
        "oracle": (oracle_acc, oracle_cost),
    }


def sweep(items: Sequence[Item]) -> List[Point]:
    base_acc, base_cost = baselines(items)["always_strong"]
    taus = [-math.inf] + sorted({i.score for i in items}) + [math.inf]
    pts = []
    for t in taus:
        acc, cost, frac = evaluate(items, t)
        pts.append(Point(t, frac, acc, cost, acc - base_acc, 1 - cost / base_cost))
    return pts


def select_tau(points: Sequence[Point], max_acc_loss: float = 0.02) -> float:
    """Largest cost saving whose accuracy loss vs always-strong is within tolerance.
    Falls back to -inf (always strong, i.e. router off) if nothing qualifies."""
    ok = [p for p in points if p.acc_delta >= -max_acc_loss - 1e-12]
    if not ok:
        return -math.inf
    best = max(ok, key=lambda p: (p.cost_cut, -abs(p.tau) if math.isfinite(p.tau) else 0))
    return best.tau


def cross_validated(items: Sequence[Item], k: int = 5, seed: int = 0,
                    max_acc_loss: float = 0.02) -> Tuple[List[Item], List[float], List[bool]]:
    """Out-of-fold routing: tau for each fold is picked on the other folds only."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    chosen: Dict[int, bool] = {}
    taus = []
    for f in range(k):
        held = set(folds[f])
        train = [items[i] for i in idx if i not in held]
        tau = select_tau(sweep(train), max_acc_loss)
        taus.append(tau)
        for i in folds[f]:
            chosen[i] = items[i].score >= tau
    return list(items), taus, [chosen[i] for i in range(len(items))]


def _policy_stats(items: Sequence[Item], use_strong: Sequence[bool]) -> Tuple[float, float, float, float]:
    n = len(items)
    acc = sum((i.strong_acc if s else i.cheap_acc) for i, s in zip(items, use_strong)) / n
    cost = sum(i.overhead + (i.strong_cost if s else i.cheap_cost) for i, s in zip(items, use_strong)) / n
    base_acc = sum(i.strong_acc for i in items) / n
    base_cost = sum(i.strong_cost for i in items) / n
    return acc, cost, acc - base_acc, 1 - cost / base_cost


def bootstrap(items: Sequence[Item], use_strong: Sequence[bool], n_boot: int = 2000,
              seed: int = 0) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(seed)
    n = len(items)
    d_acc, d_cost = [], []
    for _ in range(n_boot):
        s = [rng.randrange(n) for _ in range(n)]
        _, _, da, cc = _policy_stats([items[i] for i in s], [use_strong[i] for i in s])
        d_acc.append(da)
        d_cost.append(cc)

    def ci(v):
        v = sorted(v)
        return v[int(0.025 * n_boot)], v[int(0.975 * n_boot) - 1]
    return {"acc_delta": ci(d_acc), "cost_cut": ci(d_cost)}


# ---------- IO ----------
def build_items(questions: List[dict], results: List[dict], scores: Dict[str, Tuple[float, float]],
                cheap: str, strong: str, metric: str, cost_field: str) -> List[Item]:
    by: Dict[Tuple[str, str], dict] = {(r["qid"], r["setup"]): r for r in results}
    items = []
    for q in questions:
        qid = q["id"]
        if (qid, cheap) in by and (qid, strong) in by and qid in scores:
            c, s = by[(qid, cheap)], by[(qid, strong)]
            sc, oh = scores[qid]
            items.append(Item(qid, sc, oh, float(c[metric]), float(c[cost_field]),
                              float(s[metric]), float(s[cost_field])))
    return items


def score_questions(router, questions: List[dict], cost_field: str) -> Dict[str, Tuple[float, float]]:
    out = {}
    for q in questions:
        d = router.score(q["question"])
        out[q["id"]] = (d.difficulty, d.usage.total_tokens if cost_field == "tokens" else d.usage.usd)
    return out


def demo_items(n: int = 200, seed: int = 0) -> List[Item]:
    """SYNTHETIC. Only used to check that the pipeline and plot work. Not a result."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        hard = rng.random() < 0.4
        score = min(1, max(0, rng.gauss(0.72 if hard else 0.28, 0.17)))
        cheap_acc = float(rng.random() < (0.35 if hard else 0.78))
        strong_acc = float(rng.random() < (0.72 if hard else 0.82))
        items.append(Item(f"d{i}", score, 0.0, cheap_acc, rng.gauss(2500, 300),
                          strong_acc, rng.gauss(18000, 3000)))
    return items


# ---------- report ----------
def report(items: List[Item], cost_unit: str, k: int, max_acc_loss: float, title: str,
           out_dir: Optional[Path], synthetic: bool = False) -> str:
    base = baselines(items)
    pts = sweep(items)
    _, taus, use_strong = cross_validated(items, k, 0, max_acc_loss)
    acc, cost, d_acc, cut = _policy_stats(items, use_strong)
    ci = bootstrap(items, use_strong)
    final_tau = select_tau(pts, max_acc_loss)
    n = len(items)
    f = lambda t: "off" if t == -math.inf else ("all-cheap" if t == math.inf else f"{t:.2f}")
    ok_cost, ok_acc = cut >= 0.30, d_acc > -0.02
    lines = [f"# {title}", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DEMO DATA. Not a result. Run with real R7 outputs.**", ""]
    lines += [
        f"{n} questions, cost unit = {cost_unit}, router overhead included.", "",
        "| Policy | Accuracy | Cost/query | % routed to agent |", "|---|---:|---:|---:|",
        f"| Always cheap (S2) | {base['always_cheap'][0]:.3f} | {base['always_cheap'][1]:.0f} | 0% |",
        f"| Always strong (S4, router off) | {base['always_strong'][0]:.3f} | {base['always_strong'][1]:.0f} | 100% |",
        f"| **Router ({k}-fold CV)** | **{acc:.3f}** | **{cost:.0f}** | {100*sum(use_strong)/n:.0f}% |",
        f"| Oracle (upper bound) | {base['oracle'][0]:.3f} | {base['oracle'][1]:.0f} | n/a |", "",
        f"**Router vs always-strong:** accuracy {100*d_acc:+.1f} pts "
        f"(95% CI {100*ci['acc_delta'][0]:+.1f} to {100*ci['acc_delta'][1]:+.1f}), "
        f"cost {100*cut:.0f}% lower (95% CI {100*ci['cost_cut'][0]:.0f}% to {100*ci['cost_cut'][1]:.0f}%).", "",
        f"Target (cost cut >= 30% and accuracy loss < 2 pts): "
        f"{'MET' if ok_cost and ok_acc else 'NOT MET'} "
        f"(cost {'ok' if ok_cost else 'short'}, accuracy {'ok' if ok_acc else 'short'}). "
        f"Point estimates only; read them with the intervals above.", "",
        f"Per-fold tau: {', '.join(f(t) for t in taus)}. Tau on all data, for deployment: {f(final_tau)}.",
    ]
    md = "\n".join(lines)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "router_curve.md").write_text(md, encoding="utf-8")
        with open(out_dir / "router_curve.csv", "w") as fh:
            fh.write("tau,frac_agentic,accuracy,cost,acc_delta,cost_cut\n")
            for p in pts:
                fh.write(f"{p.tau},{p.frac_agentic:.4f},{p.acc:.4f},{p.cost:.2f},{p.acc_delta:.4f},{p.cost_cut:.4f}\n")
        _plot(pts, base, (cost, acc), cost_unit, out_dir / "router_curve.png", synthetic)
    return md


def _plot(pts, base, cv_point, cost_unit, path, synthetic):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([p.cost for p in pts], [p.acc for p in pts], "-o", ms=2, lw=1, label="router (tau sweep)")
    for name, (a, c) in base.items():
        ax.scatter([c], [a], s=60, marker="s" if name != "oracle" else "*", label=name.replace("_", " "))
    ax.scatter([cv_point[0]], [cv_point[1]], s=90, marker="D", c="k", label="router, CV-chosen tau")
    ax.set_xlabel(f"mean cost per query ({cost_unit})")
    ax.set_ylabel("accuracy")
    ax.set_title("Router cost-accuracy curve" + ("  [SYNTHETIC DEMO]" if synthetic else ""))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions"), ap.add_argument("--results")
    ap.add_argument("--cheap", default="S2"), ap.add_argument("--strong", default="S4")
    ap.add_argument("--metric", default="judge")
    ap.add_argument("--cost", default="tokens", choices=["tokens", "usd"])
    ap.add_argument("--method", default="heuristic", choices=["heuristic", "llm", "hybrid"])
    ap.add_argument("--papers", default="papers.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-acc-loss", type=float, default=0.02)
    ap.add_argument("--out", default="reports/r3")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    if a.demo:
        items = demo_items()
        print(report(items, a.cost, a.folds, a.max_acc_loss, "R3 router curve (demo)", out, synthetic=True))
        return

    from .features import PaperIndex
    from .router import Router
    questions = json.load(open(a.questions, encoding="utf-8"))
    results = [json.loads(l) for l in open(a.results, encoding="utf-8") if l.strip()]
    llm = None
    if a.method != "heuristic":
        from .llm import OpenAICompatibleClient
        llm = OpenAICompatibleClient()
    router = Router(llm=llm, index=PaperIndex.from_csv(a.papers), method=a.method)
    scores = score_questions(router, questions, a.cost)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(scores, open(out / "router_scores.json", "w"), indent=1)
    items = build_items(questions, results, scores, a.cheap, a.strong, a.metric, a.cost)
    if not items:
        raise SystemExit("No question has results for both setups; check --cheap/--strong/--metric.")
    print(report(items, a.cost, a.folds, a.max_acc_loss, f"R3 router curve ({a.cheap} vs {a.strong}, {a.method})", out))


if __name__ == "__main__":
    main()
