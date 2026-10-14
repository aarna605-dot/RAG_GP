"""R3 tests. No models, no network: everything runs against MockLLM.
Run: pytest tests/test_router.py"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.router import (LoopState, MockLLM, PaperIndex, Planner, Route, Router, StopRules,
                        per_worker_budget, route_and_plan, should_stop, trivial_plan)
from src.router.curve import Item, cross_validated, select_tau, sweep
from src.router.planner import validate

ROOT = Path(__file__).resolve().parent.parent
PAPERS = [
    {"arxiv_id": "2401.15884", "title": "Corrective Retrieval Augmented Generation"},
    {"arxiv_id": "2403.14403", "title": "Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs"},
    {"arxiv_id": "2501.00879", "title": "TrustRAG: Enhancing Robustness and Trustworthiness in RAG"},
    {"arxiv_id": "2501.00001", "title": "Another Retrieval Augmented Generation"},  # would also be "RAG"
]


@pytest.fixture
def index():
    return PaperIndex(PAPERS)


# ---- features / index ----
def test_index_finds_aliases_titles_and_ids(index):
    assert index.mentioned("What is CRAG?") == {"2401.15884"}
    assert index.mentioned("Explain Adaptive RAG routing") == {"2403.14403"}
    assert index.mentioned("Compare TrustRAG and CRAG") == {"2501.00879", "2401.15884"}
    assert "2401.15884" in index.mentioned("see arXiv:2401.15884")


def test_generic_alias_not_matched(index):
    assert index.mentioned("How does RAG work?") == set()


# ---- router ----
def test_heuristic_routes_easy_single_and_hard_agentic(index):
    r = Router(index=index, method="heuristic")
    assert r.route("What is CRAG?").route is Route.SINGLE
    assert r.route("Compare the defence in TrustRAG with the correction loop in CRAG.").route is Route.AGENTIC


def test_heuristic_has_zero_llm_cost(index):
    d = Router(index=index, method="heuristic").route("What is CRAG?")
    assert d.usage.calls == 0 and d.usage.total_tokens == 0


def test_hybrid_calls_llm_only_in_uncertain_band(index):
    llm = MockLLM(['{"label": "C", "reason": "x"}'])
    r = Router(llm=llm, index=index, method="hybrid")
    r.route("What is CRAG?")                                           # clearly easy
    r.route("Compare TrustRAG with CRAG and Adaptive-RAG.")            # clearly hard
    assert len(llm.calls) == 0
    d = r.route("How do listwise rankers differ from pointwise ones?")  # compare cue, no paper -> 0.5
    assert len(llm.calls) == 1
    assert d.route is Route.AGENTIC and d.usage.calls == 1 and d.method == "hybrid-llm"


def test_llm_label_b_keeps_uncertain_question_single(index):
    llm = MockLLM(['{"label": "B"}'])
    d = Router(llm=llm, index=index, method="hybrid").route("How do listwise rankers differ from pointwise ones?")
    assert d.route is Route.SINGLE


def test_unparseable_llm_falls_back_to_heuristic(index):
    llm = MockLLM(["I think it is hard."])
    d = Router(llm=llm, index=index, method="llm").route("What is CRAG?")
    assert d.method == "heuristic"  # label unusable -> heuristic score kept
    assert any("unparseable" in w for w in d.rationale)


def test_direct_route_only_when_enabled_and_no_corpus_entity(index):
    llm = MockLLM(['{"label": "A"}'])
    q = "What is a transformer?"
    assert Router(llm=llm, index=index, method="llm", allow_direct=False).route(q).route is Route.SINGLE
    assert Router(llm=llm, index=index, method="llm", allow_direct=True).route(q).route is Route.DIRECT
    # a named corpus paper is never answered closed-book
    assert Router(llm=llm, index=index, method="llm", allow_direct=True).route("What is CRAG?").route is not Route.DIRECT


def test_tau_moves_the_decision(index):
    q = "How do listwise rankers differ from pointwise ones?"  # score ~0.5
    assert Router(index=index, method="heuristic", tau=0.4).route(q).route is Route.AGENTIC
    assert Router(index=index, method="heuristic", tau=0.9).route(q).route is Route.SINGLE


def test_llm_method_requires_client():
    with pytest.raises(ValueError):
        Router(method="llm")


# ---- planner ----
GOOD = json.dumps({"sub_questions": [
    {"id": "q1", "text": "What defence does TrustRAG use against poisoned context?", "depends_on": []},
    {"id": "q2", "text": "How does CRAG's correction loop handle low-quality documents?", "depends_on": []},
]})


def test_planner_parses_and_waves(index):
    p = Planner(MockLLM(["```json\n" + GOOD + "\n```"]), index).plan("Compare TrustRAG and CRAG defences")
    assert not p.fallback and len(p.sub_questions) == 2
    assert [[s.id for s in w] for w in p.waves()] == [["q1", "q2"]]
    assert p.sub_questions[0].paper_hints == ["2501.00879"]
    assert p.usage.calls == 1


def test_dependent_subquestion_goes_to_later_wave_and_resolves():
    raw = {"sub_questions": [
        {"id": "q1", "text": "Which paper introduced the verifier?", "depends_on": []},
        {"id": "q2", "text": "What does {q1} report about cost?", "depends_on": ["q1"]}]}
    subs, _ = validate(raw, 4)
    from src.router import Plan
    plan = Plan("x", subs)
    assert [[s.id for s in w] for w in plan.waves()] == [["q1"], ["q2"]]
    assert subs[1].resolve({"q1": "CRAG"}) == "What does CRAG report about cost?"


def test_cycle_triggers_fallback_after_retry():
    cyc = json.dumps({"sub_questions": [
        {"id": "q1", "text": "a {q2}", "depends_on": ["q2"]}, {"id": "q2", "text": "b {q1}", "depends_on": ["q1"]}]})
    llm = MockLLM([cyc])
    p = Planner(llm, max_retries=1).plan("hard question")
    assert p.fallback and len(llm.calls) == 2 and p.usage.calls == 2
    assert p.sub_questions[0].text == "hard question"


def test_garbage_output_falls_back():
    p = Planner(MockLLM(["sorry, I can't"]), max_retries=0).plan("q?")
    assert p.fallback and len(p.sub_questions) == 1


def test_truncates_dedupes_and_drops_dangling_deps():
    raw = {"sub_questions": [{"id": f"q{i}", "text": f"distinct question {i}", "depends_on": []} for i in range(1, 8)]
           + [{"id": "q1", "text": "dup id"}]}
    subs, notes = validate(raw, 4)
    assert len(subs) == 4 and any("truncated" in n for n in notes)
    subs, notes = validate({"sub_questions": [{"id": "q1", "text": "a", "depends_on": ["q9"]}]}, 4)
    assert subs[0].depends_on == []


def test_route_and_plan_only_agentic_pays_for_planner(index):
    llm = MockLLM([GOOD])
    r, pl = Router(index=index, method="heuristic"), Planner(llm, index)
    d, plan = route_and_plan("What is CRAG?", r, pl)
    assert d.route is Route.SINGLE and len(plan.sub_questions) == 1 and plan.usage.calls == 0 and not llm.calls
    d, plan = route_and_plan("Compare TrustRAG with CRAG.", r, pl)
    assert d.route is Route.AGENTIC and plan.usage.calls == 1 and len(llm.calls) == 1


def test_trace_is_json_serialisable(index):
    d, plan = route_and_plan("Compare TrustRAG with CRAG.", Router(index=index, method="heuristic"),
                             Planner(MockLLM([GOOD]), index))
    json.dumps(d.to_trace()), json.dumps(plan.to_trace())


# ---- stop rules ----
def test_stop_order_budget_before_steps_before_sufficient():
    rules = StopRules(max_steps=3, token_budget=1000)
    s = LoopState()
    assert not should_stop(rules, s).stop
    s.record_step(["a", "b"], 1200, sufficient=True)
    assert should_stop(rules, s).reason == "budget"
    s = LoopState(); s.record_step(["a"], 10); s.record_step(["b"], 10); s.record_step(["c"], 10)
    assert should_stop(rules, s).reason == "max_steps"
    s = LoopState(); s.record_step(["a", "b"], 10, sufficient=True)
    assert should_stop(rules, s).reason == "sufficient"


def test_sufficient_ignored_without_min_evidence():
    s = LoopState(); s.record_step(["a"], 10, sufficient=True)
    assert not should_stop(StopRules(min_evidence_chunks=2), s).stop


def test_saturation_after_patience_empty_steps():
    rules = StopRules(saturation_patience=2)
    s = LoopState(); s.record_step(["a"], 1); s.record_step(["a"], 1)
    assert not should_stop(rules, s).stop           # only one empty step so far
    s.record_step(["a"], 1)
    assert should_stop(rules, s).reason == "saturation"


def test_per_worker_budget_reserves_synthesis():
    assert per_worker_budget(StopRules(token_budget=10_000, synthesis_reserve=0.3), 2) == 3500


# ---- curve ----
def _items():
    # easy questions: cheap is as good; hard questions: only strong is right
    easy = [Item(f"e{i}", 0.2, 0, 1, 1000, 1, 10000) for i in range(10)]
    hard = [Item(f"h{i}", 0.8, 0, 0, 1000, 1, 10000) for i in range(10)]
    return easy + hard


def test_select_tau_finds_the_separating_threshold():
    pts = sweep(_items())
    tau = select_tau(pts, 0.0)
    best = next(p for p in pts if p.tau == tau)
    assert best.acc_delta == 0 and best.cost_cut == pytest.approx(1 - (0.5 * 1000 + 0.5 * 10000) / 10000)


def test_router_overhead_reduces_savings():
    items = _items()
    for it in items:
        it.overhead = 500
    tau = select_tau(sweep(items), 0.0)
    best = next(p for p in sweep(items) if p.tau == tau)
    assert best.cost_cut == pytest.approx(1 - (5500 + 500) / 10000)


def test_cross_validation_returns_one_decision_per_item():
    items = _items()
    _, taus, use = cross_validated(items, k=4)
    assert len(taus) == 4 and len(use) == len(items)


# ---- cache ----
def test_cached_llm_second_call_is_free_and_persistent(tmp_path):
    from src.router import CachedLLM
    inner = MockLLM(["first", "second"])
    path = str(tmp_path / "c.jsonl")
    c = CachedLLM(inner, path, model_tag="m")
    a = c.complete("sys", "q")
    b = c.complete("sys", "q")
    assert a.text == b.text == "first" and c.hits == 1 and c.misses == 1 and len(inner.calls) == 1
    assert b.usage.total_tokens == a.usage.total_tokens          # recorded usage is replayed
    c2 = CachedLLM(MockLLM(["other"]), path, model_tag="m")        # new process, same file
    assert c2.complete("sys", "q").text == "first" and c2.misses == 0
    assert CachedLLM(MockLLM(["other"]), path, model_tag="different").complete("sys", "q").text == "other"
