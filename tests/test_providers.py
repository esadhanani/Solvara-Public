import io
import json
import unittest
from unittest.mock import patch

from solvara.providers import OllamaProvider
from solvara.router import Router, RoutingError


class OllamaTests(unittest.IsolatedAsyncioTestCase):
    async def test_payload_and_duration(self):
        payload = {"response": "local result", "done": True, "total_duration": 20_000_000}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            result = await Router([OllamaProvider("local", "test-model")]).route("test prompt", "private-tenant")
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(json.loads(request.data), {"model": "test-model", "prompt": "test prompt", "stream": False})
        self.assertFalse(result.selected.result.simulated)
        self.assertEqual(result.selected.result.backend_duration_ms, 20)
        self.assertNotIn("private-tenant", request.data.decode())

    async def test_invalid_json_is_failure(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"not json")):
            with self.assertRaises(RoutingError):
                await Router([OllamaProvider("local", "m")]).route("a", "t")

    async def test_incomplete_or_error_or_empty_response_fails(self):
        for payload in [{"response": "partial", "done": False}, {"error": "missing model"},
                        {"response": "", "done": True}, [], {"response": 123, "done": True}]:
            with self.subTest(payload=payload):
                with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
                    with self.assertRaises(RoutingError):
                        await Router([OllamaProvider("local", "m")]).route("a", "t")

    async def test_oversized_response_fails(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"x" * (4 * 1024 * 1024 + 1))):
            with self.assertRaises(RoutingError):
                await Router([OllamaProvider("local", "m")]).route("a", "t")


if __name__ == "__main__":
    unittest.main()
