# Morphantic API Endpoint Test Guide (PowerShell + curl)

This guide provides copy‑pasteable PowerShell commands (using `curl.exe`) to exercise every public endpoint on the Morphantic API at `https://api.morphantic.com`. It avoids any localhost defaults and uses the new Morphantic naming (no “AEA”).

Notes

- Requires: PowerShell 5+ (or PowerShell 7), `curl.exe` available on PATH.
- Replace placeholder values (email, password, callback URL) before running.
- For privacy: Do NOT upload datasets to the cloud API. TNaS endpoints are for local server usage only.

Terminology and Flow

- Morphantic API Server: The hosted orchestrator at `https://api.morphantic.com`. You call it for auth, jobs, optimize, batch evaluate, etc.
- Client Callback Server: A tiny HTTP endpoint that YOU host (local worker, app route, or serverless) which evaluates candidates. The Morphantic API calls this URL with POST `{"solution":[...]}` and expects metrics JSON back. This keeps your logic/data private.
- Flow (Optimize/Batch): You → Morphantic API Server → Client Callback Server → Morphantic API Server → You (results).

## 0) Setup (variables)

```powershell
# Base API URL (production)
$BASE = "https://api.morphantic.com"

# Test credentials (use unique email per run to avoid collisions)
$EMAIL = "test@gmail.com"
$PASS  = "password!"
```

## 0.5) Start your Client Callback (quick start)

Option A — SDK LocalWorker (simple, no extra deps)

```powershell
# Create a minimal callback server on http://127.0.0.1:8081/callback
$code = @'
from morphantic_core.worker import LocalWorker
import time, numpy as np

def metrics(x: np.ndarray) -> dict:
    # TODO: implement your metrics; names must match your Optimize request
    x0 = float(x[0]) if x.size else 0.0
    return {"loss": abs(x0)}  # example single-objective metric

with LocalWorker(lambda arr: metrics(arr), host="127.0.0.1", port=8081) as w:
    print("Callback URL:", w.url, flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
'@
Set-Content -Path client_callback.py -Value $code -Encoding utf8
python .\client_callback.py
# Expose publicly (pick one):
# cloudflared tunnel --url http://127.0.0.1:8081
# ngrok http 8081
# Then set: $Callback = "https://<your-public-host>/callback"
```

Option B — FastAPI (if you prefer a framework)

```powershell
python -m pip install fastapi uvicorn
$code = @'
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/callback")
async def cb(req: Request):
    d = await req.json()
    x = d.get("solution", [])
    x0 = float(x[0]) if x else 0.0
    # Return metrics; names must match your Optimize request objectives
    return {"loss": abs(x0)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
'@
Set-Content -Path client_callback_fastapi.py -Value $code -Encoding utf8
python .\client_callback_fastapi.py
# Expose publicly (pick one):
# cloudflared tunnel --url http://127.0.0.1:8081
# ngrok http 8081
# Then set: $Callback = "https://<your-public-host>/callback"
```

Quick local test (before exposing):

```powershell
curl.exe -sS -X POST "http://127.0.0.1:8081/callback" -H "Content-Type: application/json" -d '{"solution":[0.25]}'
```

## 1) Auth

Signup (idempotent for new emails):

```powershell
curl.exe -sS -X POST "$BASE/v1/auth/signup" `
  -H "Content-Type: application/json" `
  -d "{`"email`":`"$EMAIL`",`"password`":`"$PASS`",`"first_name`":`"API`",`"last_name`":`"Tester`"}"
```

Login and capture tokens (copy‑paste one‑liners):

```powershell
# Option A — Invoke-RestMethod (simplest, robust)
$login=Invoke-RestMethod -Method POST -Uri "$BASE/v1/auth/login" -ContentType 'application/json' -Body (@{email=$EMAIL;password=$PASS}|ConvertTo-Json -Compress);$TOKEN=$login.access_token;$REFRESH=$login.refresh_token;$Headers=@{Authorization="Bearer $TOKEN"}

# Option B — curl.exe with stdin (robust across PowerShell versions)
$body=@{email=$EMAIL;password=$PASS}|ConvertTo-Json -Compress;$loginRaw=$body|curl.exe -sS -X POST "$BASE/v1/auth/login" -H "Content-Type: application/json" --data-binary @-;$login=$loginRaw|ConvertFrom-Json;$TOKEN=$login.access_token;$REFRESH=$login.refresh_token;$Headers=@{Authorization="Bearer $TOKEN"}

# Option C — curl.exe with a temp file (works everywhere)
$body=@{email=$EMAIL;password=$PASS}|ConvertTo-Json -Compress;Set-Content -Path login.json -Value $body -Encoding utf8;$loginRaw=curl.exe -sS -X POST "$BASE/v1/auth/login" -H "Content-Type: application/json" --data-binary "@login.json";$login=$loginRaw|ConvertFrom-Json;$TOKEN=$login.access_token;$REFRESH=$login.refresh_token;$Headers=@{Authorization="Bearer $TOKEN"}
```

Refresh access token:

```powershell
curl.exe -sS -X POST "$BASE/v1/auth/refresh" `
  -H "Content-Type: application/json" `
  -d "{`"refresh_token`":`"$REFRESH`"}"
```

Get/update profile:

```powershell
# GET /v1/auth/me
curl.exe -sS "$BASE/v1/auth/me" -H "Authorization: Bearer $TOKEN"

# PUT /v1/auth/me
curl.exe -sS -X PUT "$BASE/v1/auth/me" `
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" `
  -d "{`"first_name`":`"New`",`"last_name`":`"Name`"}"
```

Change password (optional):

```powershell
curl.exe -sS -X POST "$BASE/v1/auth/change-password" `
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" `
  -d "{`"current_password`":`"$PASS`",`"new_password`":`"EvenStronger!456`"}"
```

## 2) API Keys (JWT required)

Create API key:

```powershell
$keyRaw = curl.exe -sS -X POST "$BASE/v1/api-keys" `
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" `
  -d "{`"current_password`":`"$PASS`",`"name`":`"CI Key`",`"expires_days`":30,`"permissions`": [`"optimize`",`"read_results`"]}"
$key    = $keyRaw | ConvertFrom-Json
$APIKEY = $key.api_key
$KEYID  = $key.id
```

List/revoke:

```powershell
curl.exe -sS "$BASE/v1/api-keys" -H "Authorization: Bearer $TOKEN"

curl.exe -sS -X DELETE "$BASE/v1/api-keys/$KEYID" -H "Authorization: Bearer $TOKEN"
```

## 3) System and Jobs

Health & usage stats:

```powershell
curl.exe -sS "$BASE/health"

curl.exe -sS "$BASE/v1/usage/stats" -H "Authorization: Bearer $TOKEN"
```

Jobs & results:

```powershell
curl.exe -sS "$BASE/v1/jobs" -H "Authorization: Bearer $TOKEN"

# For a known job ID
# curl.exe -sS "$BASE/v1/results/<job_id>" -H "Authorization: Bearer $TOKEN"
```

## 4) Optimize (SO/MO) — requires a public callback URL

Prepare a publicly reachable Client Callback URL for your objective (do not use localhost for the cloud API). Then:

Single‑Objective request body:

```powershell
$Callback = "https://your-callback-host.example.com/callback"  # Client Callback Server URL (reachable from Morphantic API)
$optSO = @{
  callback_url = $Callback
  dimension    = 3
  bounds       = @(-5.0, 5.0)
  objectives   = @(@{name="loss"; weight=1.0; baseline=10.0; target=0.1; direction="min"})
  config       = @{pop_size=24; max_generations=10; n_islands=4; seed=42}
  modules      = @("turbo")
  mode         = "so"
  constraints  = @{ hard=@(@{name="collisions"; op="=="; value=0}); soft=@(@{name="mw"; op="<="; value=550; penalty=3.0}) }
} | ConvertTo-Json -Depth 8

$jobSO  = $optSO | curl.exe -sS -X POST "$BASE/v1/optimize" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data-binary @- | ConvertFrom-Json  # ← call to Morphantic API Server
$JOB_ID = $jobSO.job_id

# Poll results
for ($i=0; $i -lt 60; $i++) {
  $res = curl.exe -sS "$BASE/v1/results/$JOB_ID" -H "Authorization: Bearer $TOKEN" | ConvertFrom-Json
  if ($res.status -in @("completed","failed")) { $res | ConvertTo-Json -Depth 8; break }
  Start-Sleep -Seconds 2
}
```

Multi‑Objective (MO):

```powershell
$optMO = @{
  callback_url = $Callback
  dimension    = 1
  bounds       = @(-1.0, 1.0)
  objectives   = @(
    @{name="a"; weight=0.5; baseline=0.0; target=1.0; direction="max"},
    @{name="b"; weight=0.5; baseline=0.0; target=1.0; direction="max"}
  )
  config       = @{pop_size=24; max_generations=10; n_islands=4; seed=42}
  modules      = @("turbo")
  mode         = "mo"
} | ConvertTo-Json -Depth 8

$jobMO  = $optMO | curl.exe -sS -X POST "$BASE/v1/optimize" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data-binary @- | ConvertFrom-Json  # ← call to Morphantic API Server
$JOBM   = $jobMO.job_id
for ($i=0; $i -lt 60; $i++) {
  $res = curl.exe -sS "$BASE/v1/results/$JOBM" -H "Authorization: Bearer $TOKEN" | ConvertFrom-Json
  if ($res.status -in @("completed","failed")) { $res | ConvertTo-Json -Depth 8; break }
  Start-Sleep -Seconds 2
}
```

## 5) Batch Evaluate — vectorized callback

```powershell
$be = @{ callback_url=$Callback; X=@(@(1,2,3), @(5,5)); timeout_s=60 } | ConvertTo-Json -Depth 6
$be | curl.exe -sS -X POST "$BASE/v1/batch_evaluate" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data-binary @-  # ← call to Morphantic API Server; it will POST to your Client Callback URL
```

## 6) TNaS (local server only; do not use with remote cloud if data is confidential)

If and only if your server is running locally and can read local paths:

Register dataset (local path is read on the server host):

```powershell
$ds = @{ name="Demo"; smiles_col="smiles"; label_col="y"; source="path"; path="C:\\path\\to\\file.csv" } | ConvertTo-Json
$dsResp = $ds | curl.exe -sS -X POST "$BASE/tnas/datasets" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data-binary @- | ConvertFrom-Json
$DATASET_ID = $dsResp.dataset_id
```

Run TNaS:

```powershell
$runBody = @{
  dataset_id = $DATASET_ID
  k          = 12
  seed       = 123
  test_frac  = 0.2
  strategy   = "morphantic"
  mode       = "scalar"
  weights    = @{ activity=0.7; diversity=0.3 }
  budget     = 200
} | ConvertTo-Json -Depth 6

$run   = $runBody | curl.exe -sS -X POST "$BASE/tnas/runs" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data-binary @- | ConvertFrom-Json
$RUNID = $run.run_id

curl.exe -sS "$BASE/tnas/runs/$RUNID" -H "Authorization: Bearer $TOKEN"
# Optional artifact (if present)
curl.exe -sS "$BASE/tnas/runs/$RUNID/artifacts/run.json" -H "Authorization: Bearer $TOKEN"
```

A/B leaderboard (local-only):

```powershell
$abBody = @{ dataset_id=$DATASET_ID; k=12; seed=123; budget=200; lanes=@("morphantic","optuna") } | ConvertTo-Json
$abBody | curl.exe -sS -X POST "$BASE/tnas/ab" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data-binary @-
```

## 7) API Key variant (optional)

If you prefer API keys for certain calls:

```powershell
$KeyHeaders = @{ 'X-API-Key' = $APIKEY }
# Example
curl.exe -sS "$BASE/v1/jobs" -H "X-API-Key: $APIKEY"
```

## 8) Pass/Fail Checklist

- Auth: signup → login → refresh → me (get/put) → change‑password.
- API keys: create → list → revoke; confirm permissions enforced on `/v1/optimize`.
- System: `/health` returns status; `/v1/usage/stats` shape OK.
- Optimize (SO): job accepted; results include `best_solution`, `final_metrics`, `mode:"so"`.
- Optimize (MO): results include `front` (and optional `hv2d` when 2D is configured server‑side).
- Batch evaluate: returns `results` array matching your callback.
- TNaS (local‑only): dataset → run → summary → artifact; A/B returns leaderboard.

---

Tip: For privacy‑preserving workflows, keep all data local. Expose only a metrics callback URL to the API (via the SDK `LocalWorker` or your own HTTPS endpoint). Do not post data or file paths to a remote server.
