import asyncio
import unittest

from solvara.providers import SyntheticProvider, synthetic_providers
from solvara.router import Policy, ProviderResult, Router, RoutingError


class Stub:
    def __init__(self, name="stub", outputs=None):
        self.name = name
        self.outputs = outputs or [ProviderResult("ok", 0.0, 0.5, True)]
        self.calls = []

    async def generate(self, prompt, tenant):
        self.calls.append((prompt, tenant))
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        if isinstance(output, Exception):
            raise output
        return output


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_latency_selects_fast_provider(self):
        result = await Router(synthetic_providers()).route("a", "t")
        self.assertEqual(result.selected.provider, "fast")

    async def test_cost_selects_economy(self):
        result = await Router(synthetic_providers()).route("a", "t", Policy(goal="cost"))
        self.assertEqual(result.selected.provider, "economy")

    async def test_quality_selects_quality(self):
        result = await Router(synthetic_providers()).route("a", "t", Policy(goal="quality"))
        self.assertEqual(result.selected.provider, "quality")

    async def test_quality_constraint_rejects_fast(self):
        result = await Router(synthetic_providers()).route("a", "t", Policy(min_quality_score=0.9))
        self.assertEqual(result.selected.provider, "quality")

    async def test_cost_constraint_rejects_expensive(self):
        result = await Router(synthetic_providers()).route("a", "t", Policy(goal="quality", max_result_cost=0.002))
        self.assertEqual(result.selected.provider, "economy")

    async def test_no_constraint_compliant_result(self):
        with self.assertRaises(RoutingError) as caught:
            await Router(synthetic_providers()).route("a", "t", Policy(max_result_cost=0))
        self.assertTrue(all(c.status == "success" for c in caught.exception.candidates))

    async def test_unrelated_prompts_do_not_reuse_prior_output(self):
        router = Router(synthetic_providers())
        first = await router.route("investment memo", "t", Policy(goal="cost"))
        second = await router.route("train timetable", "t", Policy(goal="cost"))
        self.assertNotEqual(first.selected.result.text, second.selected.result.text)

    async def test_goal_change_is_applied_each_request(self):
        router = Router(synthetic_providers())
        first = await router.route("same", "t", Policy(goal="cost"))
        second = await router.route("same", "t", Policy(goal="quality"))
        self.assertNotEqual(first.selected.provider, second.selected.provider)

    async def test_concurrent_tenants_are_separate(self):
        router = Router(synthetic_providers())
        a, b = await asyncio.gather(router.route("same", "a"), router.route("same", "b"))
        self.assertNotEqual(a.selected.result.text, b.selected.result.text)

    async def test_retry_recovers_transient_error(self):
        provider = Stub(outputs=[ConnectionError("private error"), ProviderResult("ok", 0, 1, True)])
        result = await Router([provider]).route("a", "t", Policy(retries=1))
        self.assertEqual(len(result.selected.attempts), 2)
        self.assertEqual(result.selected.attempts[0].error_type, "ConnectionError")
        self.assertNotIn("private error", str(result.to_dict()))

    async def test_retry_limit_is_enforced(self):
        provider = Stub(outputs=[ConnectionError()])
        with self.assertRaises(RoutingError):
            await Router([provider]).route("a", "t", Policy(retries=2))
        self.assertEqual(len(provider.calls), 3)

    async def test_error_falls_back_to_successful_provider(self):
        result = await Router(synthetic_providers(fail_fast=True)).route("a", "t")
        self.assertEqual(result.selected.provider, "economy")
        self.assertEqual(result.candidates[0].status, "error")

    async def test_attempt_timeout_keeps_alternative(self):
        result = await Router(synthetic_providers(slow_fast=True)).route(
            "a", "t", Policy(attempt_timeout_s=0.06))
        self.assertEqual(result.candidates[0].status, "attempt_timeout")
        self.assertEqual(result.selected.provider, "economy")

    async def test_request_timeout_retains_finished_candidate(self):
        result = await Router(synthetic_providers(slow_fast=True)).route(
            "a", "t", Policy(request_timeout_s=0.07))
        self.assertEqual(result.candidates[0].status, "request_timeout")
        self.assertEqual(result.selected.provider, "economy")

    async def test_total_timeout_reports_failure(self):
        provider = SyntheticProvider("slow", 0.2, 0, 1)
        with self.assertRaises(RoutingError) as caught:
            await Router([provider]).route("a", "t", Policy(request_timeout_s=0.01))
        self.assertEqual(caught.exception.candidates[0].status, "request_timeout")

    async def test_parent_cancellation_is_propagated(self):
        router = Router([SyntheticProvider("slow", 0.3, 0, 1)])
        task = asyncio.create_task(router.route("a", "t"))
        await asyncio.sleep(0.005)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_invalid_outputs_fail_closed(self):
        invalid = [None, "text", ProviderResult("", 0, 0, True),
                   ProviderResult("ok", float("nan"), 0, True),
                   ProviderResult("ok", -1, 0, True),
                   ProviderResult("ok", 0, 1.1, True),
                   ProviderResult("ok", 0, 0.5, "yes"),
                   ProviderResult("ok", 0, 0.5, True, -1)]
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(RoutingError):
                await Router([Stub(outputs=[output])]).route("a", "t")

    async def test_whole_request_includes_slowest_candidate(self):
        result = await Router(synthetic_providers()).route("a", "t")
        self.assertGreater(result.request_wall_ms, result.selected.wall_ms)
        self.assertGreaterEqual(result.request_wall_ms, max(c.wall_ms for c in result.candidates))
        self.assertIsNotNone(result.selected.result.backend_duration_ms)

    async def test_observed_cost_includes_every_success(self):
        result = await Router(synthetic_providers()).route("a", "t", Policy(goal="cost"))
        self.assertAlmostEqual(result.observed_success_cost, 0.017)
        self.assertGreater(result.observed_success_cost, result.selected.result.estimated_cost)

    async def test_deterministic_quality_tie_break(self):
        result = await Router([Stub(name="z"), Stub(name="a")]).route("a", "t", Policy(goal="quality"))
        self.assertEqual(result.selected.provider, "a")

    async def test_blank_input_rejected_before_provider_call(self):
        provider = Stub()
        for prompt, tenant in [("", "a"), ("a", " "), (None, "a")]:
            with self.assertRaises(ValueError):
                await Router([provider]).route(prompt, tenant)
        self.assertEqual(provider.calls, [])


class ValidationTests(unittest.TestCase):
    def test_invalid_policy_rejected(self):
        cases = [dict(goal="mystery"), dict(retries=-1), dict(retries=6), dict(retries=True),
                 dict(attempt_timeout_s=0), dict(request_timeout_s=float("inf")),
                 dict(min_quality_score=float("nan")), dict(min_quality_score=2),
                 dict(max_result_cost=-1)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Policy(**kwargs)

    def test_empty_or_duplicate_providers_rejected(self):
        for providers in ([], [Stub(), Stub()], [Stub(name="")]):
            with self.assertRaises(ValueError):
                Router(providers)


if __name__ == "__main__":
    unittest.main()
