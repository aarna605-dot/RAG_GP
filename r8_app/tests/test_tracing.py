import json

from app import tracing
from app.agents import DemoAgent


async def _final_answer():
    async for event in DemoAgent().stream("test question"):
        if event.type == "final":
            return event.answer


async def test_log_run_writes_jsonl():
    answer = await _final_answer()
    tracing.log_run(answer)
    row = json.loads(tracing.LOG_PATH.read_text().splitlines()[0])
    assert row["id"] == answer.id and row["cost"]["input_tokens"] > 0


def test_langsmith_runs_have_a_parent_and_one_child_per_step():
    import asyncio

    answer = asyncio.run(_final_answer())
    runs = tracing._langsmith_runs(answer, "proj")
    assert runs[0]["name"] == f"ask-{answer.setup}" and runs[0]["inputs"]["question"] == "test question"
    assert len(runs) == 1 + len(answer.trace) and all(r["parent_run_id"] == runs[0]["id"] for r in runs[1:])


def test_langsmith_export_posts_with_key_and_never_raises(monkeypatch):
    import asyncio

    import httpx

    answer = asyncio.run(_final_answer())
    seen = []

    class Client(httpx.Client):
        def __init__(self, **kw):
            super().__init__(transport=httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(200, json={}))[1]))

    monkeypatch.setattr(tracing.httpx, "Client", Client)
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    tracing._send_to_langsmith(answer)
    assert len(seen) == 1 + len(answer.trace) and seen[0].headers["x-api-key"] == "ls-key"
    assert str(seen[0].url).endswith("/runs")

    monkeypatch.setattr(tracing.httpx, "Client", lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
    tracing._send_to_langsmith(answer)  # must swallow the failure
