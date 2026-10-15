"""FastAPI backend: one entry point shared by the web app and the evaluation harness."""
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import store
from app.agents import Agent, default_setup, get_agent, list_setups
from app.errors import AgentError
from app.schemas import AnswerObject, AskRequest, Feedback, StreamEvent
from app.tracing import log_run

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Ask the Papers: Agentic RAG")


def _resolve_agent(req: AskRequest) -> Agent:
    try:
        return get_agent(req.setup)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _events(agent: Agent, question: str) -> AsyncIterator[StreamEvent]:
    async for event in agent.stream(question):
        if event.type == "final" and event.answer:
            log_run(event.answer)
        yield event


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerObject)
async def ask(req: AskRequest) -> AnswerObject:
    """Blocking call: returns the final answer object. Used by the eval harness."""
    agent = _resolve_agent(req)
    final: AnswerObject | None = None
    try:
        async for event in _events(agent, req.question):
            if event.type == "final":
                final = event.answer
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if final is None:
        raise HTTPException(status_code=500, detail="Agent produced no final answer")
    return final


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-sent events: live `step` events, then one `final` event (or one `error`)."""
    agent = _resolve_agent(req)  # resolved before streaming so a bad setup returns a clean 400

    async def sse() -> AsyncIterator[str]:
        try:
            async for event in _events(agent, req.question):
                yield f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"
        except AgentError as exc:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/setups")
async def setups() -> dict:
    return {"default": default_setup(), "setups": list_setups()}


@app.get("/api/history")
async def history(limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    return store.history(limit)


@app.get("/api/runs/{run_id}", response_model=AnswerObject)
async def get_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/api/feedback")
async def feedback(body: Feedback) -> dict[str, str]:
    if store.get_run(body.run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    store.add_feedback(body.run_id, body.rating)
    return {"status": "saved"}


@app.get("/api/stats")
async def stats() -> dict:
    return store.stats()


@app.get("/api/results")
async def results() -> dict:
    """Evaluation table written by R7 to runs/results.json (a JSON list, one row per setup)."""
    path = Path(os.getenv("RESULTS_FILE", "runs/results.json"))
    if not path.exists():
        return {"available": False, "rows": []}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="results.json is not valid JSON") from exc
    if not isinstance(rows, list):
        raise HTTPException(status_code=500, detail="results.json must be a list of rows")
    return {"available": True, "rows": rows}


# Must come last so it does not shadow the routes above.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
