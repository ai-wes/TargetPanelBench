# Morphantic Engine Server — FastAPI

Production API for Morphantic optimization with JWT/API key auth, MongoDB persistence, and TNaS endpoints.

## Contents

- Overview and architecture
- Run locally (MongoDB required)
- Environment variables
- Authentication (JWT, API keys)
- Core endpoints
- TNaS endpoints
- Errors and responses

## Overview

The server exposes `/v1/optimize` for single and multi‑objective runs via a secure callback to the client’s metrics function. It also provides batched evaluation, job tracking, user/account flows, API keys, and Target Narrowing as a Service (TNaS) endpoints.

## Run locally

With Docker Compose:

```bash
docker compose -f docker-compose.e2e.yml up --build
```

Manual:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="morphantic"
cd sdk/morphantic_engine_server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health:

```bash
curl http://localhost:8000/health
```

## Environment

- `MONGODB_URI` (or `MONGO_URI`): connection string
- `MONGODB_DB` (or `MONGO_DB`): DB name (default `morphantic`)
- `TNAS_STORE_DIR`, `TNAS_AB_STORE_DIR`, `TNAS_SHORTLIST_STORE`: artifact locations

## Authentication

Signup and login:

```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"Str0ng!Pass!123","first_name":"You","last_name":"User"}'

curl -X POST http://localhost:8000/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"Str0ng!Pass!123"}'
```

Use the `access_token` as a Bearer token. API keys are created with `/v1/api-keys` (JWT‑protected) and used via `X-API-Key` header.

## Core endpoints

Health:

- `GET /health` → `{status,timestamp}`

Usage stats:

- `GET /v1/usage/stats` (JWT or API key)

Jobs:

- `GET /v1/jobs` → list user jobs
- `GET /v1/results/{job_id}` → job status/result

Optimize (SO/MO):

- `POST /v1/optimize` (JWT or API key with `optimize` permission)

Request (SO example):

```json
{
  "callback_url": "http://127.0.0.1:8081/callback",
  "dimension": 3,
  "bounds": [-5.0, 5.0],
  "objectives": [
    {"name": "loss", "weight": 1.0, "baseline": 10.0, "target": 0.1, "direction": "min"}
  ],
  "config": {"pop_size": 32, "max_generations": 20, "n_islands": 4, "seed": 42},
  "modules": ["turbo"],
  "mode": "so",
  "constraints": {
    "hard": [{"name":"collisions","op":"==","value":0}],
    "soft": [{"name":"mw","op":"<=","value":550,"penalty":3.0}]
  }
}
```

Response (accepted): `{job_id}` then poll `/v1/results/{job_id}`. When completed, `result` contains `{best_solution,best_fitness,final_metrics,mode}` and for MO may include `{front,hv2d}`.

Batch evaluate:

- `POST /v1/batch_evaluate` with `{callback_url, X:[[...]], timeout_s?}` → `{"results":[{...},...]}`

## TNaS endpoints

Datasets:

- `POST /tnas/datasets` → `{dataset_id, sha256?, rows?, smiles_col, label_col}`

Runs:

- `POST /tnas/runs` with `{dataset_id,k,seed,test_frac,strategy,mode,weights?,constraints?,modules?,budget}` → `{run_id,...}`; poll `GET /tnas/runs/{id}` for result and artifacts. Use `strategy:"morphantic"` (formerly `aea`).
- `GET /tnas/runs/{id}` → summary and pointers
- `GET /tnas/runs/{id}/artifacts/{name}` → stream artifact

A/B and workflows:

- `POST /tnas/ab` with `{dataset_id,k,seed?,test_frac?,budget?,lanes:["morphantic","optuna",...]}` → `{leaderboard,artifacts}`
- `POST /tnas/feedback` with `{run_id,labels:[{index,y_true}]}` → updated selection + metrics
- `POST /tnas/shortlist` with `{dataset_id,k,budget,explore_bias}` → quick shortlist

Notes:

- Current TNaS baseline uses dependency‑light metrics (hit_rate + diversity proxy) and logs constraint violations. Upgrade to RDKit/QSAR is available by integrating the `experiments/target_nas` stack.

## Errors

- `400` invalid input, `401/403` auth/permissions, `404` not found
- `422` constraint/operator issues; `504` callback timeout
