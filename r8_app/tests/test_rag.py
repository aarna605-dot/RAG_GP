import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import agents
from app.errors import AgentError
from app.llm import LLMResult, OpenAICompatibleLLM, llm_from_env
from app.main import app
from app.rag import (
    BaselinesLoader,
    RagAgent,
    build_prompt,
    build_rag_agents,
    cited_ids,
    extractive_answer,
    normalize_citations,
)


def chunk(n, title="Corrective Retrieval Augmented Generation", paper="2401.15884", text=None):
    return {
        "chunk_id": f"{paper}_{n:04d}",
        "paper_id": paper,
        "title": title,
        "section": "Abstract",
        "header": f"{title} > Abstract",
        "text": text or (
            "CRAG uses a lightweight retrieval evaluator to judge document quality. "
            "A decompose-then-recompose step removes irrelevant content from retrieved documents."
        ),
        "score": 0.9,
    }


class FakeBaselines:
    def __init__(self, passages):
        self.passages, self.calls = passages, []

    def s1_vanilla_retrieval(self, query, top_k=5):
        self.calls.append(("s1", query, top_k))
        return self.passages[:top_k]

    def s2_tuned_retrieval(self, query, top_k=5, candidate_k=25):
        self.calls.append(("s2", query, top_k, candidate_k))
        return self.passages[:top_k]


class FakeLoader:
    def __init__(self, baselines):
        self._b = baselines

    def get(self):
        return self._b


class FakeLLM:
    model = "fake-model"

    def __init__(self, text="CRAG judges retrieval quality [1, 2]. It also filters noise [2][9].", fail=False):
        self.text, self.fail, self.prompts = text, fail, []

    def complete(self, system, user):
        self.prompts.append((system, user))
        if self.fail:
            raise AgentError("The LLM request failed with status 429: rate limited")
        return LLMResult(self.text, input_tokens=1000, output_tokens=50)


def make_agent(passages, llm=None, mode="s1", **kw):
    b = FakeBaselines(passages)
    retrieve = (lambda bl, q, k: bl.s1_vanilla_retrieval(q, top_k=k)) if mode == "s1" else (lambda bl, q, k: bl.s2_tuned_retrieval(q, top_k=k))
    return RagAgent("S1" if mode == "s1" else "S2", "Dense search, top 5", retrieve, FakeLoader(b), llm, **kw), b


async def run(agent, q="How does CRAG handle bad retrieval?"):
    events = [e async for e in agent.stream(q)]
    return events[:-1], events[-1].answer


# ---------- helpers ----------

def test_normalize_and_cited_ids():
    assert normalize_citations("a [1, 2] b [3]") == "a [1][2] b [3]"
    assert cited_ids("x [2] y [1][2] z [9] [0]", n_passages=3) == [2, 1]  # in order, deduped, in range


def test_prompt_numbers_passages_and_contains_question():
    p = build_prompt("What is CRAG?", [chunk(0), chunk(1, title="Other")])
    assert "[1] Corrective Retrieval Augmented Generation > Abstract" in p and "[2] Other > Abstract" in p
    assert p.endswith("Question: What is CRAG?\nAnswer:")


def test_extractive_answer_picks_matching_sentence_with_marker():
    out = extractive_answer("How does CRAG judge document quality?", [chunk(0)])
    assert "[1]" in out and "retrieval evaluator" in out


# ---------- the agent ----------

async def test_agent_with_llm_builds_answer_object():
    llm = FakeLLM()
    agent, b = make_agent([chunk(0), chunk(1, title="Adaptive-RAG", paper="2403.14403")], llm, price_in_per_m=1.0, price_out_per_m=2.0)
    steps, ans = await run(agent)
    assert [e.step.kind for e in steps] == ["search", "answer"]
    assert ans.setup == "S1" and ans.answer.startswith("CRAG judges retrieval quality [1][2].")
    assert [c.id for c in ans.citations] == [1, 2]  # [9] does not exist and is dropped
    assert ans.citations[1].url == "https://arxiv.org/abs/2403.14403" and ans.citations[0].section == "Abstract"
    assert (ans.cost.input_tokens, ans.cost.output_tokens) == (1000, 50)
    assert ans.cost.usd == pytest.approx((1000 * 1.0 + 50 * 2.0) / 1e6)
    assert b.calls == [("s1", "How does CRAG handle bad retrieval?", 5)]
    assert "ONLY the numbered passages" in llm.prompts[0][0]


async def test_s2_uses_the_tuned_retrieval():
    agent, b = make_agent([chunk(0)], FakeLLM("Yes [1]."), mode="s2")
    await run(agent)
    assert b.calls[0][0] == "s2"


async def test_agent_without_llm_falls_back_and_says_so():
    agent, _ = make_agent([chunk(0)], llm=None)
    steps, ans = await run(agent, "How does CRAG judge document quality?")
    assert "No answer model is configured" in ans.answer and ans.citations and ans.cost.input_tokens == 0
    assert "No LLM configured" in steps[-1].step.message


async def test_no_passages_means_no_llm_call_and_no_citations():
    llm = FakeLLM(fail=True)  # would raise if it were called
    agent, _ = make_agent([], llm)
    _, ans = await run(agent)
    assert ans.citations == [] and "no passages" in ans.answer.lower()


async def test_llm_failure_raises_agent_error():
    agent, _ = make_agent([chunk(0)], FakeLLM(fail=True))
    with pytest.raises(AgentError, match="429"):
        await run(agent)


# ---------- the API with a registered RAG agent ----------

def test_api_uses_registered_s1_and_reports_errors(monkeypatch):
    ok, _ = make_agent([chunk(0)], FakeLLM("Good [1]."))
    bad, _ = make_agent([chunk(0)], FakeLLM(fail=True))
    monkeypatch.setitem(agents.REGISTRY, "S1", ok)
    monkeypatch.setitem(agents.REGISTRY, "S2", bad)
    client = TestClient(app)

    by_id = {s["id"]: s for s in client.get("/api/setups").json()["setups"]}
    assert by_id["S1"]["available"]

    data = client.post("/ask", json={"question": "What is CRAG?", "setup": "S1"}).json()
    assert data["setup"] == "S1" and data["citations"][0]["paper_title"].startswith("Corrective")

    r = client.post("/ask", json={"question": "What is CRAG?", "setup": "S2"})
    assert r.status_code == 502 and "429" in r.json()["detail"]

    with client.stream("POST", "/ask/stream", json={"question": "What is CRAG?", "setup": "S2"}) as s:
        lines = [ln for ln in s.iter_lines() if ln]
    assert "event: error" in lines
    assert "429" in json.loads(lines[lines.index("event: error") + 1].removeprefix("data: "))["message"]


# ---------- wiring and config ----------

def test_build_rag_agents_needs_the_r2_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_REPO_PATH", str(tmp_path))
    assert build_rag_agents() == {}
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "retriever.py").write_text("")
    (tmp_path / "indices" / "bm25").mkdir(parents=True)
    (tmp_path / "indices" / "lancedb").mkdir(parents=True)
    assert set(build_rag_agents()) == {"S1", "S2"}


def test_loader_explains_missing_libraries(tmp_path, monkeypatch):
    import sys

    for name in [n for n in sys.modules if n == 'src' or n.startswith('src.')]:
        monkeypatch.delitem(sys.modules, name)  # other tests may have loaded the real package
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "retriever.py").write_text("import library_that_is_not_installed\n")
    (tmp_path / "src" / "baselines.py").write_text("")
    with pytest.raises(AgentError, match="requirements-rag.txt"):
        BaselinesLoader(tmp_path).get()


def test_llm_from_env(monkeypatch):
    assert llm_from_env() is None
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert llm_from_env() is None  # model missing
    monkeypatch.setenv("LLM_MODEL", "m")
    assert llm_from_env().model == "m"


def test_openai_compatible_client_request_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hi [1] "}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    llm = OpenAICompatibleLLM("secret", "m1", base_url="https://example.test/v1/", transport=httpx.MockTransport(handler))
    res = llm.complete("sys", "usr")
    assert (res.text, res.input_tokens, res.output_tokens) == ("Hi [1]", 7, 3)
    assert seen["url"] == "https://example.test/v1/chat/completions" and seen["auth"] == "Bearer secret"
    assert seen["body"]["temperature"] == 0 and seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_compatible_client_errors_never_leak_the_key():
    llm = OpenAICompatibleLLM("secret", "m1", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key secret")))
    with pytest.raises(AgentError) as exc:
        llm.complete("s", "u")
    assert "401" in str(exc.value) and "secret" not in str(exc.value)


def test_repo_is_found_in_the_parent_folder(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "retriever.py").write_text("")
    (tmp_path / "indices" / "bm25").mkdir(parents=True)
    (tmp_path / "indices" / "lancedb").mkdir(parents=True)
    (tmp_path / "r8-app").mkdir()
    monkeypatch.delenv("RAG_REPO_PATH", raising=False)
    monkeypatch.chdir(tmp_path / "r8-app")
    assert set(build_rag_agents()) == {"S1", "S2"}
    (tmp_path / "elsewhere" / "deep").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "elsewhere" / "deep")  # neither here nor in the parent
    assert build_rag_agents() == {}


def test_llm_client_retries_rate_limits(monkeypatch):
    import httpx

    from app.llm import OpenAICompatibleLLM

    monkeypatch.setattr("app.llm.time.sleep", lambda s: None)
    codes = iter([429, 503, 200])

    def handler(req):
        code = next(codes)
        body = {"choices": [{"message": {"content": "ok"}}], "usage": {}} if code == 200 else {}
        return httpx.Response(code, json=body)

    llm = OpenAICompatibleLLM("k", "m", transport=httpx.MockTransport(handler))
    assert llm.complete("s", "u").text == "ok"
