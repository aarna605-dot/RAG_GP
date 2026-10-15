"""R3: router + planner + stop rules."""
from .schemas import Route, RouteDecision, SubQuestion, Plan, Usage
from .features import PaperIndex
from .router import Router
from .planner import Planner, route_and_plan, trivial_plan
from .stop_rules import StopRules, LoopState, should_stop, per_worker_budget, can_retry_verification
from .llm import LLMClient, MockLLM, OpenAICompatibleClient, CachedLLM

__all__ = [
    "Route", "RouteDecision", "SubQuestion", "Plan", "Usage", "PaperIndex", "Router", "Planner",
    "route_and_plan", "trivial_plan", "StopRules", "LoopState", "should_stop", "per_worker_budget",
    "can_retry_verification", "LLMClient", "MockLLM", "OpenAICompatibleClient", "CachedLLM",
]
