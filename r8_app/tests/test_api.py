import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _ask(q="When does agentic RAG help?"):
    return client.post("/ask", json={"question": q}).json()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_web_app_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "Ask the Papers" in r.text


def test_ask_returns_answer_object():
    data = _ask()
    assert data["id"] and data["answer"] and data["citations"] and data["trace"]
    assert data["cost"]["input_tokens"] > 0


def test_demo_answer_follows_the_question():
    assert "router" in _ask("How much does a router save?")["answer"].lower()
    assert "injection" in _ask("Can an attack hijack the agent?")["answer"].lower()


def test_stream_emits_steps_then_one_final():
    with client.stream("POST", "/ask/stream", json={"question": "Compare S1 and S4"}) as r:
        events = [ln.removeprefix("event:").strip() for ln in r.iter_lines() if ln.startswith("event:")]
    assert events[-1] == "final" and events.count("final") == 1 and "step" in events


def test_unknown_or_unregistered_setup_is_rejected():
    for path in ("/ask", "/ask/stream"):
        assert client.post(path, json={"question": "hello there", "setup": "S4"}).status_code == 400


def test_short_question_is_rejected():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_setups_lists_registered_and_pending():
    data = client.get("/api/setups").json()
    by_id = {s["id"]: s for s in data["setups"]}
    assert data["default"] == "mock" and by_id["mock"]["available"] and not by_id["S4"]["available"]


def test_history_run_lookup_and_stats():
    assert client.get("/api/history").json() == []
    first = _ask("first question here")
    second = _ask("second question here")
    hist = client.get("/api/history").json()
    assert [h["id"] for h in hist] == [second["id"], first["id"]]  # newest first
    assert client.get(f"/api/runs/{first['id']}").json()["question"] == "first question here"
    assert client.get("/api/runs/nope").status_code == 404
    s = client.get("/api/stats").json()
    assert s["answers"] == 2 and s["tokens"] > 0 and s["usd"] > 0


def test_feedback_is_saved_and_counted():
    run = _ask()
    assert client.post("/api/feedback", json={"run_id": run["id"], "rating": "up"}).status_code == 200
    assert client.post("/api/feedback", json={"run_id": run["id"], "rating": "down"}).status_code == 200
    assert client.get("/api/stats").json()["thumbs_down"] == 1  # latest rating wins
    assert client.post("/api/feedback", json={"run_id": "nope", "rating": "up"}).status_code == 404
    assert client.post("/api/feedback", json={"run_id": run["id"], "rating": "meh"}).status_code == 422


def test_results_missing_valid_and_invalid(tmp_path, monkeypatch):
    assert client.get("/api/results").json() == {"available": False, "rows": []}
    path = tmp_path / "results.json"
    path.write_text(json.dumps([{"setup": "S1", "judge_acc": 0.41}]))
    assert client.get("/api/results").json()["rows"][0]["setup"] == "S1"
    path.write_text("{not json")
    assert client.get("/api/results").status_code == 500
