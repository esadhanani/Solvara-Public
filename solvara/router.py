"""Fan out requests, measure attempts, enforce constraints, and select a result."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

Goal = Literal["latency", "cost", "quality"]


@dataclass(frozen=True)
class ProviderResult:
    text: str
    estimated_cost: float
    quality_score: float
    simulated: bool
    # Optional duration reported by the backend, distinct from network wall time.
    backend_duration_ms: float | None = None


class Provider(Protocol):
    name: str

    async def generate(self, prompt: str, tenant: str) -> ProviderResult: ...


@dataclass(frozen=True)
class Policy:
    goal: Goal = "latency"
    attempt_timeout_s: float = 1.0
    request_timeout_s: float = 3.0
    retries: int = 0
    max_result_cost: float | None = None
    min_quality_score: float = 0.0

    def __post_init__(self) -> None:
        if self.goal not in ("latency", "cost", "quality"):
            raise ValueError("goal must be latency, cost, or quality")
        for value in (self.attempt_timeout_s, self.request_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts must be finite and positive")
        if type(self.retries) is not int or not 0 <= self.retries <= 5:
            raise ValueError("retries must be an integer between 0 and 5")
        if not math.isfinite(self.min_quality_score) or not 0 <= self.min_quality_score <= 1:
            raise ValueError("min_quality_score must be between 0 and 1")
        if self.max_result_cost is not None and (
            not math.isfinite(self.max_result_cost) or self.max_result_cost < 0
        ):
            raise ValueError("max_result_cost must be finite and nonnegative")


@dataclass(frozen=True)
class Attempt:
    number: int
    wall_ms: float
    status: str
    error_type: str | None = None


@dataclass(frozen=True)
class Candidate:
    provider: str
    result: ProviderResult | None
    attempts: tuple[Attempt, ...]
    wall_ms: float
    status: str


@dataclass(frozen=True)
class RouteResult:
    selected: Candidate
    candidates: tuple[Candidate, ...]
    request_wall_ms: float
    goal: Goal
    observed_success_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


class RoutingError(RuntimeError):
    def __init__(self, candidates: tuple[Candidate, ...], request_wall_ms: float):
        super().__init__("No successful result satisfied the policy")
        self.candidates = candidates
        self.request_wall_ms = request_wall_ms


def validate_result(result: ProviderResult) -> None:
    if not isinstance(result, ProviderResult):
        raise ValueError("provider must return ProviderResult")
    if not isinstance(result.text, str) or not result.text.strip():
        raise ValueError("provider returned empty or invalid text")
    if not math.isfinite(result.estimated_cost) or result.estimated_cost < 0:
        raise ValueError("invalid provider cost")
    if not math.isfinite(result.quality_score) or not 0 <= result.quality_score <= 1:
        raise ValueError("invalid quality score")
    if type(result.simulated) is not bool:
        raise ValueError("simulated must be a boolean")
    if result.backend_duration_ms is not None and (
        not math.isfinite(result.backend_duration_ms) or result.backend_duration_ms < 0
    ):
        raise ValueError("invalid backend duration")


class Router:
    """Evaluate all providers concurrently, then select a policy-compliant result.

    No response cache or prompt history is retained. A provider adapter must be
    cancellation-cooperative and safe for concurrent calls. Remote work might
    continue after client cancellation, so observed cost is never a spend cap.
    """

    def __init__(self, providers: list[Provider]):
        if not providers:
            raise ValueError("at least one provider is required")
        names = [p.name for p in providers]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("provider names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("provider names must be unique")
        self.providers = tuple(providers)

    async def _run(self, provider: Provider, prompt: str, tenant: str, policy: Policy) -> Candidate:
        started = time.perf_counter()
        attempts: list[Attempt] = []
        for index in range(policy.retries + 1):
            attempt_started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    provider.generate(prompt, tenant), timeout=policy.attempt_timeout_s
                )
                validate_result(result)
            except asyncio.CancelledError:
                attempts.append(Attempt(index + 1, (time.perf_counter() - attempt_started) * 1000,
                                        "request_timeout", "CancelledError"))
                return Candidate(provider.name, None, tuple(attempts),
                                 (time.perf_counter() - started) * 1000, "request_timeout")
            except Exception as exc:
                status = "attempt_timeout" if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else "error"
                attempts.append(Attempt(index + 1, (time.perf_counter() - attempt_started) * 1000,
                                        status, type(exc).__name__))
            else:
                attempts.append(Attempt(index + 1, (time.perf_counter() - attempt_started) * 1000,
                                        "success"))
                return Candidate(provider.name, result, tuple(attempts),
                                 (time.perf_counter() - started) * 1000, "success")
        return Candidate(provider.name, None, tuple(attempts),
                         (time.perf_counter() - started) * 1000, attempts[-1].status)

    async def route(self, prompt: str, tenant: str, policy: Policy | None = None) -> RouteResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(tenant, str) or not tenant.strip():
            raise ValueError("tenant must be a nonempty string")
        policy = policy or Policy()
        started = time.perf_counter()
        tasks = [asyncio.create_task(self._run(p, prompt, tenant, policy)) for p in self.providers]
        try:
            _, pending = await asyncio.wait(tasks, timeout=policy.request_timeout_s)
            for task in pending:
                task.cancel()
            candidates = tuple(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        wall_ms = (time.perf_counter() - started) * 1000
        eligible = [c for c in candidates if c.result is not None
                    and c.result.quality_score >= policy.min_quality_score
                    and (policy.max_result_cost is None
                         or c.result.estimated_cost <= policy.max_result_cost)]
        if not eligible:
            raise RoutingError(candidates, wall_ms)
        if policy.goal == "latency":
            key = lambda c: (c.wall_ms, c.provider)
        elif policy.goal == "cost":
            key = lambda c: (c.result.estimated_cost, -c.result.quality_score, c.provider)
        else:
            key = lambda c: (-c.result.quality_score, c.result.estimated_cost, c.provider)
        return RouteResult(min(eligible, key=key), candidates, wall_ms, policy.goal,
                           sum(c.result.estimated_cost for c in candidates if c.result is not None))
