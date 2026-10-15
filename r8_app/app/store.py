"""Read-side helpers over the run log: history, single runs, feedback and stats.

Runs are appended to `runs.jsonl` by `app.tracing.log_run`. Feedback goes to a
`feedback.jsonl` file next to it. Both are plain JSON lines, so R7 can read them too.
"""
import json
from pathlib import Path

from app import tracing


def _feedback_path() -> Path:
    return tracing.LOG_PATH.with_name("feedback.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip a corrupt line rather than failing the whole page
    return rows


def load_runs() -> list[dict]:
    runs = _read_jsonl(tracing.LOG_PATH)
    for i, row in enumerate(runs):
        row.setdefault("id", f"legacy-{i}")
    return runs


def history(limit: int = 50) -> list[dict]:
    items = [
        {
            "id": r["id"],
            "question": r.get("question", ""),
            "setup": r.get("setup", ""),
            "usd": r.get("cost", {}).get("usd", 0.0),
            "latency_s": r.get("cost", {}).get("latency_s", 0.0),
        }
        for r in load_runs()
    ]
    return list(reversed(items))[:limit]


def get_run(run_id: str) -> dict | None:
    return next((r for r in load_runs() if r["id"] == run_id), None)


def add_feedback(run_id: str, rating: str) -> None:
    path = _feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "rating": rating}) + "\n")


def stats() -> dict:
    runs = load_runs()
    latest_rating = {f["run_id"]: f["rating"] for f in _read_jsonl(_feedback_path())}
    known = {r["id"] for r in runs}
    ratings = [v for k, v in latest_rating.items() if k in known]
    costs = [r.get("cost", {}) for r in runs]
    n = len(runs)
    return {
        "answers": n,
        "avg_latency_s": round(sum(c.get("latency_s", 0) for c in costs) / n, 2) if n else 0.0,
        "tokens": sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in costs),
        "usd": round(sum(c.get("usd", 0) for c in costs), 5),
        "thumbs_up": ratings.count("up"),
        "thumbs_down": ratings.count("down"),
    }
