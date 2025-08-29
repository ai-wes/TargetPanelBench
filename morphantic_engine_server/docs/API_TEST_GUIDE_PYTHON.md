# Morphantic API Endpoint Test Guide (Python)

This guide shows how to exercise the Morphantic API at `https://api.morphantic.com` using pure Python (`requests`) and how to run a Client Callback Server in Python to keep your objective function and data private.

Notes

- Do not upload datasets to the cloud API. TNaS dataset endpoints are intended for local/private deployments.
- All examples target the production base URL: `https://api.morphantic.com`.
- Terminology:
  - Morphantic API Server: hosted orchestrator at `https://api.morphantic.com`.
  - Client Callback Server: your small HTTP endpoint that scores candidate solutions. The API posts `{"solution":[...]}` and expects metrics JSON back. It must be reachable from the API (public URL for the hosted service).

## 0) Prerequisites

```bash
python -m pip install requests fastapi uvicorn pyngrok websockets
```

```python
BASE = "https://api.morphantic.com"  # Morphantic API Server
EMAIL = "your_email@example.com"
PASS  = "YourStr0ngPassword!"
```

## 1) Auth (signup, login, refresh, profile)

Create `auth_demo.py`:

```python
import requests

BASE = "https://api.morphantic.com"
EMAIL = "your_email@example.com"
PASS  = "YourStr0ngPassword!"

# 1) Signup (safe to call if user already exists)
requests.post(
    f"{BASE}/v1/auth/signup",
    json={"email": EMAIL, "password": PASS, "first_name": "API", "last_name": "Tester"},
    timeout=30,
)

# 2) Login
r = requests.post(f"{BASE}/v1/auth/login", json={"email": EMAIL, "password": PASS}, timeout=30)
r.raise_for_status()
tokens = r.json()
access = tokens["access_token"]
refresh = tokens["refresh_token"]
print("Login OK; expires_in:", tokens.get("expires_in"))
headers = {"Authorization": f"Bearer {access}"}

# 3) Refresh
rr = requests.post(f"{BASE}/v1/auth/refresh", json={"refresh_token": refresh}, timeout=30)
rr.raise_for_status()
print("Refresh OK")

# 4) Profile
me = requests.get(f"{BASE}/v1/auth/me", headers=headers, timeout=30)
print("Profile:", me.json())

# Update profile (optional)
requests.put(
    f"{BASE}/v1/auth/me",
    headers=headers,
    json={"first_name": "New", "last_name": "Name"},
    timeout=30,
)
```

Run: `python auth_demo.py`

## 2) API Keys (create, list, delete)

```python
import requests

BASE = "https://api.morphantic.com"
EMAIL = "your_email@example.com"
PASS  = "YourStr0ngPassword!"

# login
tok = requests.post(f"{BASE}/v1/auth/login", json={"email": EMAIL, "password": PASS}, timeout=30).json()
headers = {"Authorization": f"Bearer {tok['access_token']}"}

# create key
data = {
    "current_password": PASS,
    "name": "CI Key",
    "expires_days": 30,
    "permissions": ["optimize", "read_results"],
}
key = requests.post(f"{BASE}/v1/api-keys", headers=headers, json=data, timeout=30).json()
print("API key created:", key.get("api_key"))

# list
print(requests.get(f"{BASE}/v1/api-keys", headers=headers, timeout=30).json())

# delete
kid = key.get("id")
if kid:
    requests.delete(f"{BASE}/v1/api-keys/{kid}", headers=headers, timeout=30)
```

## 3) Client Callback Server (Python)

The Morphantic API calls your callback with `POST {"solution":[...]}` and expects `{<metric_name>: <float>, ...}`. Keys must match `objectives[*].name` (and any constraint names).

Option A — FastAPI (recommended)

Create `callback_fastapi.py`:

```python
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/callback")
async def callback(req: Request):
    payload = await req.json()
    x = payload.get("solution", []) or []
    x0 = float(x[0]) if x else 0.0
    # TODO: swap in your real metrics
    return {
        "loss": abs(x0),
        # For MO:
        # "a": max(0.0, 1.0 - abs(x0)),
        # "b": max(0.0, x0),
        # For constraints (if configured):
        # "collisions": 0,
        # "mw": 480.0,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
```

Start it: `python callback_fastapi.py` (serves `http://127.0.0.1:8081/callback`)

Option B — Standard library only

Create `callback_stdlib.py`:

```python
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return
    def do_POST(self):  # noqa: N802
        if self.path != "/callback":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body)
            x = data.get("solution", []) or []
            x0 = float(x[0]) if x else 0.0
            resp = json.dumps({"loss": abs(x0)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:  # noqa: BLE001
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8081), Handler).serve_forever()
```

Local test (before exposing):

```bash
curl -sS -X POST http://127.0.0.1:8081/callback -H "Content-Type: application/json" -d '{"solution":[0.25]}'
```

Expose publicly so the hosted API can reach it:

- Cloudflare Tunnel: `cloudflared tunnel --url http://127.0.0.1:8081`
- ngrok: `ngrok http 8081`
- Use the printed HTTPS URL (e.g., `https://<id>.trycloudflare.com/callback`) as your `callback_url` in the requests below.

One-file Quickstart (zero manual shell setup)

```bash
# Installs and runs a single Python script that:
# - starts the local callback
# - opens a public ngrok tunnel automatically
# - uses your API key for all calls (no JWT login)
# - runs a batch-evaluate smoke test
# - runs a single-objective optimize job and prints results
python -m pip install requests fastapi uvicorn pyngrok
set MORPH_API_KEY=your_api_key_here   # use export on bash
python sdk/morphantic_core/examples/quickstart_onefile.py
```

If you prefer manual control, continue with the sections below.

WebSocket Quickstart (no tunnels)

```bash
# Start a WS worker that connects out to the API and serves evals
python -m pip install websockets numpy requests
set MORPH_API_KEY=your_api_key_here
python sdk/morphantic_core/examples/quickstart_ws.py
# It will connect and serve. In another terminal, you can run the optimize demo
# (optimize currently uses HTTP callback; WS is supported for batch_evaluate now and can be extended to optimize later.)
```

Now what (connect end‑to‑end)

- Set your public callback URL (HTTPS) from cloudflared/ngrok; it must end with `/callback`.
- Run a smoke test via Batch Evaluate to confirm reachability.
- Then run the optimize demo script that polls results.

Quick smoke test (Python):

```python
import requests
BASE = "https://api.morphantic.com"
EMAIL = "you@example.com"; PASS = "YourStr0ngPassword!"
CALLBACK = "https://<your-ngrok-or-cf-host>/callback"

# Auth
requests.post(f"{BASE}/v1/auth/signup", json={"email": EMAIL, "password": PASS, "first_name": "API", "last_name": "Tester"}, timeout=30)
access = requests.post(f"{BASE}/v1/auth/login", json={"email": EMAIL, "password": PASS}, timeout=30).json()["access_token"]
headers = {"Authorization": f"Bearer {access}"}

# Batch evaluate
body = {"callback_url": CALLBACK, "X": [[0.2], [0.8]], "timeout_s": 60}
print(requests.post(f"{BASE}/v1/batch_evaluate", headers=headers, json=body, timeout=60).json())
```

## 4) Optimize (SO/MO) with polling

Create `optimize_demo.py`:

```python
import os, time, requests

BASE = "https://api.morphantic.com"  # Morphantic API Server
EMAIL = os.environ.get("MORPH_EMAIL", "your_email@example.com")
PASS  = os.environ.get("MORPH_PASS",  "YourStr0ngPassword!")
CALLBACK = os.environ.get("MORPH_CALLBACK", "https://your-public-host.example.com/callback")  # Client Callback Server URL

# Auth
requests.post(f"{BASE}/v1/auth/signup", json={"email": EMAIL, "password": PASS, "first_name": "API", "last_name": "Tester"}, timeout=30)
access = requests.post(f"{BASE}/v1/auth/login", json={"email": EMAIL, "password": PASS}, timeout=30).json()["access_token"]
headers = {"Authorization": f"Bearer {access}"}

# Single-Objective optimize
so = {
    "callback_url": CALLBACK,
    "dimension": 1,
    "bounds": [-1.0, 1.0],
    "objectives": [
        {"name": "loss", "weight": 1.0, "baseline": 1.0, "target": 0.0, "direction": "min"}
    ],
    "config": {"pop_size": 12, "max_generations": 5, "n_islands": 2},
    "mode": "so",
}
resp = requests.post(f"{BASE}/v1/optimize", headers=headers, json=so, timeout=30)
resp.raise_for_status()
job_id = resp.json()["job_id"]
print("SO job:", job_id)

# Poll
for _ in range(120):
    r = requests.get(f"{BASE}/v1/results/{job_id}", headers=headers, timeout=30)
    data = r.json()
    if data.get("status") in ("completed", "failed"):
        print("SO result:", data)
        break
    time.sleep(1)

# Multi-Objective optimize (metric names must match your callback response)
mo = {
    "callback_url": CALLBACK,
    "dimension": 1,
    "bounds": [-1.0, 1.0],
    "objectives": [
        {"name": "a", "weight": 0.5, "baseline": 0.0, "target": 1.0, "direction": "max"},
        {"name": "b", "weight": 0.5, "baseline": 0.0, "target": 1.0, "direction": "max"},
    ],
    "config": {"pop_size": 12, "max_generations": 5, "n_islands": 2},
    "mode": "mo",
}
resp = requests.post(f"{BASE}/v1/optimize", headers=headers, json=mo, timeout=30)
job_id = resp.json()["job_id"]
print("MO job:", job_id)
for _ in range(120):
    r = requests.get(f"{BASE}/v1/results/{job_id}", headers=headers, timeout=30)
    data = r.json()
    if data.get("status") in ("completed", "failed"):
        print("MO result:", data)
        break
    time.sleep(1)
```

Run with API key: `MORPH_API_KEY=your_api_key_here MORPH_CALLBACK=https://<your-public-host>/callback python sdk/morphantic_core/examples/optimize_demo.py`

## 5) Batch Evaluate

```python
import os, requests
BASE = "https://api.morphantic.com"
EMAIL = os.environ.get("MORPH_EMAIL", "your_email@example.com")
PASS  = os.environ.get("MORPH_PASS",  "YourStr0ngPassword!")
CALLBACK = os.environ.get("MORPH_CALLBACK", "https://your-public-host.example.com/callback")

access = requests.post(f"{BASE}/v1/auth/login", json={"email": EMAIL, "password": PASS}, timeout=30).json()["access_token"]
headers = {"Authorization": f"Bearer {access}"}

body = {"callback_url": CALLBACK, "X": [[1,2,3],[5,5]], "timeout_s": 60}
print(requests.post(f"{BASE}/v1/batch_evaluate", headers=headers, json=body, timeout=60).json())
```

## 6) TNaS (local/private servers only)

Do not register local file paths or datasets with the hosted API. TNaS endpoints that reference paths are intended for private deployments where the server can read local paths.

## 7) SDK alternative (optional)

The SDK can spin a local callback for you. Example sketch:

```python
from morphantic_core import MorphanticClient, UserCredentials
from morphantic_core.worker import LocalWorker

client = MorphanticClient(base_url="https://api.morphantic.com")
client.login(UserCredentials(email="your_email@example.com", password="YourStr0ngPassword!"))

with LocalWorker(lambda x: {"loss": abs(float(x[0]))}) as w:
    job = client.optimize({
        "callback_url": w.url,
        "dimension": 1,
        "bounds": [-1.0, 1.0],
        "objectives": [{"name":"loss","weight":1.0,"baseline":1.0,"target":0.0,"direction":"min"}],
        "config": {"pop_size": 8, "max_generations": 3, "n_islands": 2},
        "mode": "so",
    })
    print("job:", job)
```

## 8) Troubleshooting

- 401 Unauthorized: ensure valid JWT in `Authorization: Bearer <token>`.
- 422 JSON errors: verify your request JSON matches the schema (names, types).
- 504/timeout: your callback must be reachable from the internet and respond within the configured timeout.
- Metric names must exactly match your `objectives[*].name` and any constraint names; otherwise defaults/baselines may be used, degrading results.
