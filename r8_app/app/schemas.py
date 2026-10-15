"""The shared 'answer object' contract.

Every module in the project reads or writes these models. Agree on this file
with the whole team on day 1; the API, the web app and the evaluation harness
all depend on it.
"""
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_run_id() -> str:
    return uuid4().hex[:12]


class Citation(BaseModel):
    id: int
    paper_title: str
    section: str
    passage: str
    url: str | None = None


class TraceStep(BaseModel):
    kind: Literal["plan", "search", "read", "verify", "answer"]
    message: str
    latency_ms: int = 0


class Cost(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    usd: float = 0.0


class AnswerObject(BaseModel):
    id: str = Field(default_factory=_new_run_id)
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    cost: Cost = Field(default_factory=Cost)
    setup: str = "mock"


class StreamEvent(BaseModel):
    """One message on the live stream: trace steps, then one final answer."""

    type: Literal["step", "final"]
    step: TraceStep | None = None
    answer: AnswerObject | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    setup: str | None = None  # e.g. "S1", "S4"; None uses the default agent


class Feedback(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
