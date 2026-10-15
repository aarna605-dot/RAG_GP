"""Stop rules for the search-read-reflect loop (R4 calls these every iteration).

Order matters and is deterministic so runs are reproducible at temperature 0:
  1. token budget   2. step cap   3. enough evidence   4. saturation
Budget and step cap are hard limits; (3) and (4) stop early to save cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set


@dataclass(frozen=True)
class StopRules:
    max_steps: int = 8                  # S3 cap from the brief
    max_steps_per_worker: int = 3       # S4 workers
    max_subquestions: int = 4
    token_budget: int = 30_000          # PLACEHOLDER: R7 fixes the real per-setup budget
    synthesis_reserve: float = 0.3      # share of budget kept for synthesis + verification
    saturation_patience: int = 2        # consecutive steps with no new chunks -> stop
    min_evidence_chunks: int = 2        # "sufficient" is ignored below this
    max_verifier_retries: int = 1       # "unsupported: search again" loops


@dataclass
class LoopState:
    steps: int = 0
    tokens_used: int = 0
    evidence: Set[str] = field(default_factory=set)
    new_chunks_history: List[int] = field(default_factory=list)
    sufficient: bool = False            # set from the agent's reflection step
    verifier_retries: int = 0

    def record_step(self, chunk_ids: Iterable[str], tokens: int, sufficient: bool = False) -> int:
        new = set(chunk_ids) - self.evidence
        self.evidence |= new
        self.steps += 1
        self.tokens_used += tokens
        self.new_chunks_history.append(len(new))
        self.sufficient = sufficient
        return len(new)


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str     # "budget" | "max_steps" | "sufficient" | "saturation" | "continue"


def should_stop(rules: StopRules, state: LoopState, step_cap: int = None) -> StopDecision:
    cap = step_cap if step_cap is not None else rules.max_steps
    if state.tokens_used >= rules.token_budget:
        return StopDecision(True, "budget")
    if state.steps >= cap:
        return StopDecision(True, "max_steps")
    if state.sufficient and len(state.evidence) >= rules.min_evidence_chunks:
        return StopDecision(True, "sufficient")
    n = rules.saturation_patience
    if state.steps >= n and all(c == 0 for c in state.new_chunks_history[-n:]):
        return StopDecision(True, "saturation")
    return StopDecision(False, "continue")


def per_worker_budget(rules: StopRules, n_subquestions: int) -> int:
    """Tokens each parallel worker may spend; the reserve is kept for synthesis + verifier."""
    usable = int(rules.token_budget * (1 - rules.synthesis_reserve))
    return usable // max(1, n_subquestions)


def can_retry_verification(rules: StopRules, state: LoopState) -> bool:
    return state.verifier_retries < rules.max_verifier_retries
