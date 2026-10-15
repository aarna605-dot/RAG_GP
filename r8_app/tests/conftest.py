import os

os.environ["RAG_REPO_PATH"] = "/nonexistent-rag-repo-for-tests"  # before app.agents is imported
os.environ.pop("LLM_API_KEY", None)
os.environ.pop("LLM_MODEL", None)

import pytest  # noqa: E402

from app import tracing  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    """Every test writes its run log, feedback and results into its own temp folder."""
    monkeypatch.setattr(tracing, "LOG_PATH", tmp_path / "runs.jsonl")
    monkeypatch.setenv("RESULTS_FILE", str(tmp_path / "results.json"))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
