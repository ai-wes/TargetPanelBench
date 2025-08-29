# Morphantic Core Engine API Reference (v2.0.0)

This document describes the HTTP and WebSocket interfaces exposed by the Morphantic Core API as implemented in `main.py`.

- Base URL: `http://<host>:8000`
- Run locally: `uvicorn main:app --reload`
- CORS: All origins allowed (intended for development; restrict in production).

## Authentication

Two auth modes are supported. Provide one on each request that requires auth:

- JWT access tokens: send `Authorization: Bearer <access_token>`
- API Keys: send `X-API-Key: <api_key>`

Most endpoints accept either. API Key permissions may be enforced per-route (see notes below).

Environment/config:
- `JWT_SECRET_KEY`: HMAC secret for JWT signing. If not set, a random secret is generated at boot (tokens won’t survive restarts).
- MongoDB is required for all auth and job state: set `MONGODB_URI` and optional `MONGODB_DB` (default `morphantic`).

### Auth Endpoints

- POST `/v1/auth/signup`
  - Body: `{ email, password, first_name, last_name }`
  - Validates password strength; returns user profile.
- POST `/v1/auth/login`
  - Body: `{ email, password }`
  - Returns `{ access_token, refresh_token, token_type: "bearer", expires_in }`.
- POST `/v1/auth/refresh`
  - Body: `{ refresh_token }`
  - Returns a new `{ access_token }`; reuses `refresh_token`.
- GET `/v1/auth/me`
  - Returns the current user profile.
- PUT `/v1/auth/me`
  - Body: `{ first_name?, last_name? }`
  - Updates profile fields.
- POST `/v1/auth/change-password`
  - Body: `{ current_password, new_password }`
  - Validates and updates to the new password.

### API Key Management

- POST `/v1/api-keys`
  - Auth: JWT only. Confirms `{ current_password }`.
  - Body: `{ current_password, name?, expires_days?, permissions?, rate_limit? }`
  - Returns `{ api_key, id, name, permissions, created_at, expires_at }`.
  - Note: `api_key` is shown once; store it securely.
- GET `/v1/api-keys`
  - Lists active keys for the caller.
- DELETE `/v1/api-keys/{key_id}`
  - Revokes the key.

### Usage Stats

- GET `/v1/usage/stats`
  - Returns aggregate usage statistics for the caller. Currently returns zeros until usage logging is wired.

## Optimization API

Morphantic exposes an asynchronous optimization job API backed by the archipelago engine. You provide either domain presets or explicit objectives and a callback endpoint or WebSocket for on-demand metric evaluation.

### Concepts

- Objectives: weighted targets to minimize toward (lower is better after scalarization), each defined as:
  - `ObjectiveSpec`: `{ name: str, weight: float, baseline: float, target: float, direction: "min" | "max" }`
- Constraints:
  - Hard: must satisfy; violation adds a very large penalty.
  - Soft: weighted penalty added when violated.
  - `ConstraintExpr`: `{ name: str, op: string, value: number, penalty?: number }`
  - `Constraints`: `{ hard?: ConstraintExpr[], soft?: ConstraintExpr[] }`
  - Supported `op`: `<=`, `<`, `>=`, `>`, `==`, `!=`
- Modules: plug-in behaviors that shape search dynamics. Supported names:
  - `safezone`, `drift_guard`, `turbo`, `diversity`
- Evaluation: either HTTP callback or WebSocket. The engine requests metrics for each candidate `solution` vector.

### Start Optimization

- POST `/v1/optimize` 202 Accepted
  - Auth: JWT or API Key with `optimize` permission
  - Body `OptimizationRequest`:
    - `callback_url: string` — HTTP endpoint to call with `{ solution: number[] }` and return `{ [metricName]: number }`. Ignored when `use_websocket=true`.
    - `dimension: int` — number of decision variables (1..2048).
    - `bounds: [number, number]` — lower and upper bounds for each dimension.
    - `config?: { pop_size?: int, max_generations?: int, n_islands?: int, seed?: int }`
    - `objectives?: ObjectiveSpec[]` — required if `domain` not provided.
    - `domain?: string` — optional preset; currently supports `"MorphanticBio"` (adds `diversity` module and default objectives).
    - `mode?: "so" | "mo"` — single- or multi-objective; MO returns a trivial front using weights.
    - `constraints?: Constraints` — hard and/or soft constraints applied on returned metrics.
    - `modules?: string[]` — additional module names (see above).
    - `use_websocket?: boolean` — prefer WebSocket evaluation (requires active WS connection).
  - Response: `{ job_id, message }`

Example request:

```bash
curl -X POST http://localhost:8000/v1/optimize \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "callback_url": "http://localhost:9000/score",
    "dimension": 4,
    "bounds": [-1.0, 1.0],
    "objectives": [
      {"name":"loss","weight":1.0,"baseline":1.0,"target":0.0,"direction":"min"}
    ],
    "config": {"pop_size":60, "max_generations":45, "n_islands":4},
    "constraints": {"soft":[{"name":"latency_ms","op":"<=","value":50,"penalty":2.0}]},
    "modules": ["drift_guard","diversity"]
  }'
```

Example callback server (Python/FastAPI):

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class ScoreIn(BaseModel):
    solution: list[float]

@app.post("/score")
async def score(inp: ScoreIn):
    x = np.asarray(inp.solution, float)
    return {
        "loss": float((x**2).sum()),
        "latency_ms": 12.3
    }
```

### Get Job Result

- GET `/v1/results/{job_id}`
  - Auth: JWT or API Key
  - Response:
    - `status: "queued" | "running" | "completed" | "failed"`
    - `result?` on completed:
      - `{ best_solution: number[], best_fitness: number, final_metrics: {..}, mode, front?, pick? }`
    - `error_message?` on failed

### List Jobs

- GET `/v1/jobs?limit=10&offset=0`
  - Auth: JWT or API Key
  - Response: `{ jobs: [{ id, status, domain, started_at, completed_at, result?, error_message? }, ...] }`

### Batch Evaluate

Evaluate multiple candidates via the same callback or WebSocket.

- POST `/v1/batch_evaluate`
  - Auth: JWT or API Key with `optimize` permission
  - Body: `{ callback_url: string, X: number[][], timeout_s?: int, use_websocket?: bool }`
  - Response: `{ results: Array<metrics | { error: string }> }`

## WebSocket Evaluation Channel

- Path: `GET /v1/eval/ws`
- Auth: must include `X-API-Key` header; the key’s user is bound to the connection.
- Flow: server pushes eval tasks; client replies with results.

Headers:
- `X-API-Key: <api_key>`

Server → Client messages:
- Eval task: `{ "type": "eval", "task_id": string, "solution": number[], "timeout_s": int }`
- Ping: `{ "type": "ping" }` (client may receive and should reply with `pong`)

Client → Server messages:
- Success: `{ "type": "result", "task_id": string, "metrics": { [metricName]: number } }`
- Error: `{ "type": "error", "task_id": string, "message": string }`
- Ping reply: `{ "type": "pong" }`

Notes:
- The optimization request must set `use_websocket: true` to prefer WS evaluation.
- If no WS client is connected for the user, HTTP callback is used (or errors if only WS is feasible).
- Timeouts are enforced per task on the server side.

Minimal Python client example (websockets):

```python
import asyncio, json, websockets

API_KEY = "<your key>"
URL = "ws://localhost:8000/v1/eval/ws"

async def main():
    async with websockets.connect(URL, extra_headers={"X-API-Key": API_KEY}) as ws:
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "eval":
                x = msg["solution"]
                # Compute metrics here
                metrics = {"loss": sum(v*v for v in x)}
                await ws.send(json.dumps({"type":"result","task_id":msg["task_id"],"metrics":metrics}))
            elif msg.get("type") == "ping":
                await ws.send(json.dumps({"type": "pong"}))

asyncio.run(main())
```

## TNaS (Targeted N-Selection) API

Experimental selection utilities over molecule- or item-sets using CSV datasets.

Environment/config:
- `TNAS_STORE_DIR`: directory to store run artifacts (default `artifacts/tnas`).
- `TNAS_AB_STORE_DIR`: directory to store A/B comparison artifacts (default `artifacts/tnas_ab`).

Dataset schema expectations:
- CSV with at least a `smiles` column. If `label_col` exists and is numeric, it’s used as an activity proxy for hit rate.

### Create Dataset

- POST `/tnas/datasets`
  - Auth: JWT or API Key
  - Body `TNaSDatasetIn`:
    - `name: string`
    - `smiles_col: string`
    - `label_col: string`
    - `source: "path" | "upload" | "url"` (currently path-only is read locally)
    - `path?: string`
    - `url?: string`
  - Response `TNaSDatasetOut`: `{ dataset_id, sha256?, rows?, smiles_col, label_col }`
  - Notes: when `source="path"` and `path` is readable, row count and SHA256 are recorded.

### Start Run

- POST `/tnas/runs`
  - Auth: JWT or API Key
  - Body `TNaSRunIn`:
    - `dataset_id: string`
    - `k: int` — size of the selection (default 12)
    - `seed?: int`, `test_frac?: float`, `strategy?: string`
    - `mode?: "scalar" | "mo"` — if `mo`, returns a simple Pareto front proxy
    - `weights?: { [name]: number }` — weight for activity vs diversity when scalarizing
    - `constraints?: Constraints` — optional selection constraints
    - `budget: int` — exploration budget
    - `use_websocket?: boolean`, `modules?: string[]` — reserved for parity with optimizer
  - Response `TNaSRunOut`: `{ run_id, metrics_eval, metrics_model, selection, artifacts, elapsed_s }`
  - Behavior: computes a selection by random candidate portfolios under constraints; writes `run.json` under `TNAS_STORE_DIR`.

### Get Run

- GET `/tnas/runs/{run_id}`
  - Auth: JWT or API Key
  - Response: `{ run_id, status, result?, artifacts?, created_at, completed_at?, error_message? }`

### Get Artifact

- GET `/tnas/runs/{run_id}/artifacts/{name}`
  - Auth: JWT or API Key
  - Returns the artifact file (e.g., `run.json`) as a download.

### A/B Comparison

- POST `/tnas/ab`
  - Auth: JWT or API Key
  - Body `TNaSAbIn`: `{ dataset_id, k, seed?, test_frac?, budget, lanes: string[] }`
  - Response `TNaSAbOut`: `{ leaderboard: [...], artifacts: { compare.json } }`

## Health

- GET `/health` → `{ status: "healthy", timestamp }`

## Error Handling

- Standard HTTP status codes are used:
  - 400 Bad Request: validation errors (e.g., missing required fields)
  - 401 Unauthorized: invalid/missing credentials
  - 403 Forbidden: permission or API key issues
  - 404 Not Found: missing resources (job, dataset, run, artifact)
  - 502/503: backend evaluation or WS channel issues
- Response bodies contain `{ detail: string }` for raised errors or structured error messages where noted.

## Permissions Summary (API Keys)

- `optimize`: required for `/v1/optimize` and `/v1/batch_evaluate`
- `read_results`: recommended for reading results (not strictly enforced in code for listing/reads but plan accordingly)

## Operational Notes

- Determinism: the engine seeds RNGs when `config.seed` is provided; for strict reproducibility ensure identical callback logic and environment.
- Threads: the engine library sets single-threaded math defaults for stability.
- Token lifetimes: access token ~30 minutes; refresh token ~7 days.
- Security: prefer JWT in first-party backends; use API Keys for service-to-service calls and WebSocket eval.

## Quick Start Checklist

1) Start API: `uvicorn main:app --reload`
2) Set `MONGODB_URI` (and optional `JWT_SECRET_KEY`) in environment or `.env`.
3) `POST /v1/auth/signup` → `POST /v1/auth/login` to obtain tokens.
4) Create an API key if you plan to use WebSocket evaluation.
5) Implement your scoring callback to return metrics for a candidate `solution`.
6) `POST /v1/optimize` with objectives or a supported `domain`.
7) Poll `GET /v1/results/{job_id}` until `status = completed`.

---
Generated from code in `main.py`, `auth.py`, and `mongo_store.py`. Keep this doc in sync with the code.

