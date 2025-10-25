# Solvara — Distributed LLM Inference Infrastructure

Solvara is a lightweight framework that simulates distributed inference routing for large language models.  
It explores latency-aware load balancing, message-passing, and dynamic node management across multi-GPU clusters.

## Overview
Solvara routes inference requests intelligently across a cluster of compute nodes, adapting to:
- Network latency and throughput  
- Node health and availability  
- Dynamic load conditions in real time  

This system demonstrates how distributed inference can be optimized to reduce end-to-end latency while maintaining reliability and scalability.

## Architecture

[ Request Queue ] → [ Router ] → [ Node 1 | Node 2 | Node 3 … ]
↘︎ metrics + telemetry → dashboard

Each node runs a simulated inference stub that mimics GPU utilization and response times.  
The router continuously updates weights based on observed performance to optimize routing.

## Features
- Parallel request dispatch using Python `asyncio`
- Latency regression model trained with `scikit-learn`
- Configurable cluster topology via `config.yaml`
- REST API built with `FastAPI` for submitting inference requests
- Lightweight Docker support for containerized testing

## Example Usage
```bash
python router.py --nodes 3 --batch_size 16
curl -X POST http://localhost:8000/infer -d '{"prompt": "What is Solvara?"}'

Tech Stack
	•	Python 3.9+
	•	FastAPI
	•	NumPy / Pandas / Scikit-learn
	•	Docker (optional for deployment)

Purpose

This project represents ongoing work on distributed AI inference infrastructure,
originally developed as part of internal experiments and early-stage demos for enterprise AI pilots (CloudHQ)
and applications to Y Combinator and Z Fellows.

⸻

📍 Created by Esa Dhanani at Imperial College London
🔗 unifylayer.com
