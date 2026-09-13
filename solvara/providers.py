"""Offline fixtures and an optional local Ollama adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import urllib.request
from dataclasses import dataclass

from .router import ProviderResult


@dataclass(frozen=True)
class SyntheticProvider:
    name: str
    delay_s: float
    estimated_cost: float
    quality_score: float
    fail: bool = False

    async def generate(self, prompt: str, tenant: str) -> ProviderResult:
        await asyncio.sleep(self.delay_s)
        if self.fail:
            raise ConnectionError("injected synthetic provider failure")
        digest = hashlib.sha256(json.dumps([tenant, prompt]).encode()).hexdigest()[:16]
        return ProviderResult(
            text=f"Synthetic fixture from {self.name}; input fingerprint {digest}",
            estimated_cost=self.estimated_cost,
            quality_score=self.quality_score,
            simulated=True,
            backend_duration_ms=self.delay_s * 1000,
        )


@dataclass(frozen=True)
class OllamaProvider:
    name: str
    model: str
    endpoint: str = "http://127.0.0.1:11434"
    socket_timeout_s: float = 10.0
    estimated_cost: float = 0.0
    quality_score: float = 0.0

    async def generate(self, prompt: str, tenant: str) -> ProviderResult:
        # Tenant is required by the interface but never sent to the model.
        # urllib runs in a worker thread. Cancellation stops waiting, not the
        # socket or server inference; the independent socket timeout still applies.
        return await asyncio.to_thread(self._request, prompt)

    def _request(self, prompt: str) -> ProviderResult:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        request = urllib.request.Request(
            self.endpoint.rstrip("/") + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.socket_timeout_s) as response:
            # Bound accidental or malicious response sizes to 4 MiB.
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Ollama response exceeds 4 MiB")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("error"):
            raise ValueError("Ollama returned an error or invalid response")
        if data.get("done") is not True:
            raise ValueError("Ollama response is incomplete")
        duration = data.get("total_duration")
        return ProviderResult(
            text=data.get("response"), estimated_cost=self.estimated_cost,
            quality_score=self.quality_score, simulated=False,
            backend_duration_ms=None if duration is None else duration / 1_000_000,
        )


def synthetic_providers(*, fail_fast: bool = False, slow_fast: bool = False) -> list[SyntheticProvider]:
    return [
        SyntheticProvider("fast", 0.15 if slow_fast else 0.002, 0.004, 0.65, fail_fast),
        SyntheticProvider("economy", 0.012, 0.001, 0.7),
        SyntheticProvider("quality", 0.025, 0.012, 0.95),
    ]
