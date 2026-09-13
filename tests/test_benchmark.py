import unittest

from solvara.__main__ import benchmark, percentile


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_benchmark_counts_and_labels(self):
        result = await benchmark(1)
        self.assertEqual(result["mode"], "synthetic_offline")
        self.assertEqual(result["summary"]["requests"], 9)
        self.assertEqual(result["summary"]["route_failures"], 0)
        self.assertEqual(result["summary"]["provider_failures"], 6)
        self.assertGreaterEqual(result["summary"]["request_wall_ms"]["p95"],
                                result["summary"]["request_wall_ms"]["p50"])

    async def test_invalid_repeat_count(self):
        with self.assertRaises(ValueError):
            await benchmark(0)

    def test_nearest_rank_percentile(self):
        self.assertEqual(percentile(list(range(1, 101)), 0.95), 95)
        self.assertIsNone(percentile([], 0.5))


if __name__ == "__main__":
    unittest.main()
