"""Run logging and optional LangSmith export.

Every finished answer is appended to a JSONL file, which the evaluation harness (R7)
can read for latency, tokens and cost. If LANGSMITH_API_KEY is set, the run is also sent to
LangSmith (one parent run per question, one child run per agent step). Export happens in a
background thread and is best-effort: a LangSmith failure never slows or breaks an answer.

    LANGSMITH_API_KEY    enables export
    LANGSMITH_PROJECT    optional, default "agentic-rag"
    LANGSMITH_ENDPOINT   optional, default https://api.smith.langchain.com

Uses LangSmith's REST API through httpx, so no extra library is needed.
"""
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from app.schemas import AnswerObject

logger = logging.getLogger(__name__)
LOG_PATH = Path(os.getenv("RUN_LOG", "runs/runs.jsonl"))


def _langsmith_runs(answer: AnswerObject, project: str) -> list[dict]:
    """One parent 'chain' run plus one child run per trace step, laid out on a timeline."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=answer.cost.latency_s)
    parent_id = str(uuid.uuid4())
    runs = [
        {
            "id": parent_id,
            "name": f"ask-{answer.setup}",
            "run_type": "chain",
            "session_name": project,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "inputs": {"question": answer.question},
            "outputs": {"answer": answer.answer, "citations": [c.model_dump() for c in answer.citations]},
            "extra": {
                "metadata": {
                    "setup": answer.setup,
                    "run_id": answer.id,
                    "input_tokens": answer.cost.input_tokens,
                    "output_tokens": answer.cost.output_tokens,
                    "usd": answer.cost.usd,
                }
            },
        }
    ]
    cursor = start
    for step in answer.trace:
        step_end = cursor + timedelta(milliseconds=step.latency_ms)
        runs.append(
            {
                "id": str(uuid.uuid4()),
                "parent_run_id": parent_id,
                "name": step.kind,
                "run_type": "tool",
                "session_name": project,
                "start_time": cursor.isoformat(),
                "end_time": step_end.isoformat(),
                "inputs": {"step": step.kind},
                "outputs": {"message": step.message},
            }
        )
        cursor = step_end
    return runs


def _send_to_langsmith(answer: AnswerObject) -> None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return
    base = (os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
    project = os.getenv("LANGSMITH_PROJECT") or "agentic-rag"
    try:
        with httpx.Client(timeout=15) as client:
            for run in _langsmith_runs(answer, project):
                resp = client.post(f"{base}/runs", headers={"x-api-key": api_key}, json=run)
                resp.raise_for_status()
    except Exception as exc:  # never leak the key, never break an answer
        logger.warning("LangSmith export failed: %s", type(exc).__name__)


def log_run(answer: AnswerObject) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(answer.model_dump_json() + "\n")
    if os.getenv("LANGSMITH_API_KEY"):
        threading.Thread(target=_send_to_langsmith, args=(answer,), daemon=True).start()
