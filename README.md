# Solvara

A small Python experiment for comparing model-provider responses under cost, latency, and quality-score constraints. It runs candidates concurrently, records failures and timings, and selects a result for each request.

This September 2026 rebuild replaces an earlier public README-only snapshot with runnable code and tests. It is a local prototype, not a deployed inference service. The default providers are synthetic fixtures. Their outputs, prices, and quality scores are invented test inputs, not measured model performance.

## Run

Python 3.9 or later. No third-party runtime or test dependencies.

```sh
git clone https://github.com/esadhanani/Solvara-Public.git
cd Solvara-Public
python3 -m solvara demo --goal latency
python3 -m solvara demo --goal cost
python3 -m solvara demo --goal quality
python3 -m unittest discover -s tests -v
python3 -m solvara benchmark --repeats 5 --output artifacts/benchmark.json
```

The benchmark makes 45 requests across three policies and three scenarios: healthy providers, a failed fast provider, and a slow fast provider. It reports route failures, provider failures, selected-provider counts, and p50/p95 wall time. Selection inputs are repeatable; measured times vary with machine load. The checked-in sample was produced locally using the command above and is labelled `synthetic_offline` throughout. It makes no GPU or real inference speed claim.

## How selection works

1. Validate the prompt, tenant identifier, and policy.
2. Call each configured provider concurrently. Each attempt has a timeout; retries are bounded and immediate.
3. Stop waiting at the request deadline, retain completed results, and cancel pending work.
4. Reject empty or malformed results, then filter by maximum result cost and minimum quality score.
5. Select the lowest wall time, lowest estimated cost, or highest supplied quality score. Cost and quality ties use stable provider-name ordering after the documented secondary criterion in the code.

There is no response cache, shared prompt history, model-training loop, or LLM judge. Every request supplies its own prompt and tenant identifier. The tenant value separates synthetic fixture inputs; it is not authentication or access control. Real adapters must implement their own data isolation where needed.

This is **fan-out evaluation followed by selection**, not predictive routing before inference. It waits for all candidates or the deadline and can cost more than calling one model. A fast selected candidate does not mean the whole request was fast.

Three timing fields keep that distinction visible:

| Field | Meaning |
| --- | --- |
| `backend_duration_ms` | Optional duration reported by the backend; configured delay for synthetic providers |
| `selected.wall_ms` | Elapsed client time for the selected provider, including its retries |
| `request_wall_ms` | Complete route call, including other candidates, timeouts, and selection |

`max_result_cost` filters candidate outputs. It is **not a request spending limit**. `observed_success_cost` adds the cost estimates of all successful candidates. Failed or cancelled remote work may still incur charges, which this prototype cannot observe. Quality scores are supplied by adapters or fixtures, not independently verified measures of answer correctness.

## Use from Python

```python
import asyncio
from solvara import Policy, Router
from solvara.providers import synthetic_providers

async def main():
    router = Router(synthetic_providers())
    result = await router.route(
        prompt="Summarise the supplied evidence",
        tenant="demo-team",
        policy=Policy(goal="cost", min_quality_score=0.7, retries=1),
    )
    print(result.selected.provider)
    print(result.request_wall_ms)

asyncio.run(main())
```

Implement the asynchronous `generate(prompt, tenant)` interface and return `ProviderResult` to add an adapter. Providers must tolerate concurrent calls and cooperate with cancellation. A failed route raises `RoutingError`, preserving candidate attempt statuses and elapsed time. Exception messages are excluded from reports because upstream errors can contain private inputs.

## Optional local Ollama call

With Ollama running and the chosen models already installed:

```sh
python3 -m solvara ollama --models llama3.2:1b llama3.2:3b --prompt "Explain a database index in one sentence" --timeout 30
```

The adapter sends requests to `http://127.0.0.1:11434/api/generate` by default. It validates complete responses and records Ollama's reported total duration separately from client wall time. It does not send the tenant identifier. The CLI compares latency only: Ollama does not supply answer-quality measurements or a monetary cost per request. Zero cost here means no configured API charge, not free hardware or energy.

HTTP calls use the standard library in a worker thread. Cancelling the asynchronous wait does not stop the underlying socket or remote inference. The socket timeout is independent, and process shutdown can wait for an outstanding worker. Do not treat the request deadline as hard real-time cancellation. Live Ollama inference has not been measured for this release; adapter tests use mocked HTTP responses.

## Verification and scope

The tests cover per-request policy changes, unrelated prompts, concurrent tenant inputs, timeout fallback, transient failure recovery, retry limits, cancellation, malformed output, cost and quality constraints, timing accounting, and Ollama response validation. CI runs the tests on Python 3.9, 3.11, and 3.13.

The project has no web server, credentials, persistent request logs, customer documents, or historical production results. A production version would need authenticated tenant boundaries, provider-specific billing, concurrency limits, observability, calibrated quality evaluation, and a selection policy that can choose before paying for every candidate.
