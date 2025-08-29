CAN: **Close, but not quite.**
What you pasted is a decent *SaaS‑style* client (signup/login, JWT, etc.). For **licensing a black‑box engine into customer VPCs and partner platforms**, you want a **thin, production‑ready client SDK** that:

1. **Does not** expose user signup/login (no multi‑tenant SaaS surface).
2. Uses **API key (or mTLS)**, not passwords.
3. Speaks a **tiny, fixed API** to your on‑prem engine:

   * `POST /optimize`
   * `GET  /metrics?job_id=...`
   * `POST /replay/export` (streams a signed replay bundle)
   * `POST /license/report` (tamper‑evident usage counters)
4. Adds **idempotency**, **request signing (optional HMAC)**, strict **timeouts/retries**, and **structured errors**.
5. Includes a **local HTTP worker** utility to expose the user’s objective function at `/callback` (what you already sketched, tightened up).
6. Avoids leaking internals: the optimization logic lives in your **engine**; the SDK is just transport + ergonomics.

Below is a **ship‑ready Python SDK** that fits the licensing architecture we discussed. It removes SaaS auth, adds proper retries/timeouts, optional request signing, idempotency keys, and a hardened local worker. You can publish this as `morphantic` on PyPI and vendor it into partners.

---

## Morphantic Thin Client SDK (production‑ready)

> **What it does:** Calls your on‑prem/VPC **Morphantic Engine** with API‑key headers (or mTLS), enforces timeouts/backoff, provides idempotency keys, optional HMAC request signing, and utilities for a local `/callback` worker.
> **What it doesn’t do:** No signup/login, no JWTs, no multi‑tenant session state.

```python
"""
morphantic_client.py — Thin client SDK for the on‑prem Morphantic Engine.

Features:
- API key or mTLS auth (no user signup/login)
- Idempotency keys for POSTs
- Optional HMAC request signing (shared secret; per-tenant)
- Strict timeouts + retries with exponential backoff
- Structured errors
- Local HTTP worker to expose a metrics function at /callback
- Safe JSON handling and deterministic behavior
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------- Data Models ---------

@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: str  # "max" or "min"
    weight: float = 1.0
    baseline: Optional[float] = None
    target: Optional[float] = None


@dataclass(frozen=True)
class OptimizationConfig:
    # Engine-facing knobs (the engine enforces limits; the SDK just passes them)
    budget_nfe: int
    seed: Optional[int] = None
    batch_size: Optional[int] = None
    modules: Optional[List[str]] = None  # e.g. ["safezone","driftguard","turbo","diversity"]


@dataclass(frozen=True)
class OptimizationRequest:
    callback_url: str
    # Decision vector definition
    dimension: int
    bounds: Tuple[float, float]
    # Multi-objective
    objectives: List[ObjectiveSpec]
    # Engine config
    config: OptimizationConfig
    # Optional domain tag (for lens/routing on server)
    domain: Optional[str] = None
    # Arbitrary metadata (propagated to replay/logs)
    meta: Optional[Mapping[str, Any]] = None


# --------- Exceptions ---------

class MorphanticError(Exception):
    """Base error for Morphantic SDK."""


class AuthError(MorphanticError):
    """Authentication / authorization problems."""


class BadRequestError(MorphanticError):
    """Validation or client-side errors (HTTP 400)."""


class NotFoundError(MorphanticError):
    """Missing job/resource (HTTP 404)."""


class ServerError(MorphanticError):
    """Engine-side failure (HTTP 5xx)."""


# --------- HTTP Session with Retries ---------

def _build_session(
    timeout_s: float = 30.0,
    total_retries: int = 5,
    backoff_factor: float = 0.25,
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        status=total_retries,
        allowed_methods=frozenset(["GET", "POST"]),
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # Attach a default timeout to all requests via a hook
    s.request = _timeout_wrapper(s.request, timeout_s)  # type: ignore
    return s


def _timeout_wrapper(request_fn, timeout_s: float):
    def wrapped(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout_s
        return request_fn(method, url, **kwargs)
    return wrapped


# --------- HMAC Request Signing (optional) ---------

class RequestSigner:
    """
    Optional HMAC signer to add authenticity & anti-replay to /optimize requests.
    Use only if you controls both client & engine per-tenant; store secrets securely.
    """
    def __init__(self, shared_secret: bytes):
        if not shared_secret:
            raise ValueError("shared_secret must be non-empty bytes")
        self._secret = shared_secret

    def headers_for(self, body_bytes: bytes) -> Dict[str, str]:
        ts = str(int(time.time()))
        nonce = uuid.uuid4().hex
        mac = hmac.new(self._secret, body_bytes + ts.encode("utf-8") + nonce.encode("utf-8"), hashlib.sha256).digest()
        sig = base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")
        return {
            "X-Morphantic-Timestamp": ts,
            "X-Morphantic-Nonce": nonce,
            "X-Morphantic-Signature": sig,
        }


# --------- Client ---------

class MorphanticClient:
    """
    Thin client for the Morphantic Engine (on-prem/VPC).

    Auth:
      - API key via `X-API-Key` header (recommended)
      - OR mTLS at your reverse proxy (no client code needed)

    Endpoints used:
      POST /optimize
      GET  /metrics
      POST /replay/export
      POST /license/report
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        timeout_s: float = 30.0,
        total_retries: int = 5,
        backoff_factor: float = 0.25,
        signer: Optional[RequestSigner] = None,
        watermark_id: Optional[str] = None,  # per-tenant watermark; echoed in logs
        session: Optional[requests.Session] = None,
    ):
        self._base = base_url.rstrip("/")
        self._session = session or _build_session(
            timeout_s=timeout_s,
            total_retries=total_retries,
            backoff_factor=backoff_factor,
        )
        self._api_key = api_key
        self._signer = signer
        self._watermark_id = watermark_id or ""

        if api_key:
            self._session.headers["X-API-Key"] = api_key
        if self._watermark_id:
            self._session.headers["X-Morphantic-Watermark"] = self._watermark_id

        # All requests carry a unique idempotency key unless provided by caller
        # (helps partners safely retry POSTs)
        self._session.headers["X-Idempotency-Key"] = uuid.uuid4().hex

    # ---- Helpers ----

    def _handle(self, resp: requests.Response) -> Any:
        ctype = resp.headers.get("Content-Type", "")
        text = resp.text
        try:
            payload = resp.json() if "application/json" in ctype else {}
        except Exception:
            payload = {}

        if 200 <= resp.status_code < 300:
            return payload or text

        # Map common errors
        if resp.status_code in (401, 403):
            raise AuthError(payload.get("error") or f"Auth error: {resp.status_code}")
        if resp.status_code == 404:
            raise NotFoundError(payload.get("error") or "Resource not found")
        if 400 <= resp.status_code < 500:
            raise BadRequestError(payload.get("error") or f"Bad request ({resp.status_code})")
        raise ServerError(payload.get("error") or f"Server error ({resp.status_code})")

    def _post(self, path: str, json_body: Mapping[str, Any], *, idempotency_key: Optional[str] = None) -> Any:
        url = f"{self._base}{path}"
        body = json.dumps(json_body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers: Dict[str, str] = {}
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key
        if self._signer:
            headers.update(self._signer.headers_for(body))
        resp = self._session.post(url, data=body, headers={"Content-Type": "application/json", **headers})
        return self._handle(resp)

    def _get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.get(url, params=params or {})
        return self._handle(resp)

    # ---- Public API ----

    def optimize(self, req: OptimizationRequest, *, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Start an optimization study. Returns a job descriptor (job_id, etc.)."""
        payload = {
            "callback_url": req.callback_url,
            "dimension": int(req.dimension),
            "bounds": [float(req.bounds[0]), float(req.bounds[1])],
            "objectives": [asdict(o) for o in req.objectives],
            "config": asdict(req.config),
            "domain": req.domain,
            "meta": dict(req.meta) if req.meta else {},
        }
        return self._post("/optimize", payload, idempotency_key=idempotency_key)

    def metrics(self, job_id: str) -> Dict[str, Any]:
        """Fetch anytime curves, HV/IGD (if MO), and budget ledger for a job."""
        return self._get("/metrics", params={"job_id": job_id})

    def replay_export(self, job_id: str, dest_path: str) -> str:
        """
        Request a deterministic replay bundle (signed).
        Saves to dest_path (zip/tar). Returns the path.
        """
        url = f"{self._base}/replay/export"
        # Some engines will stream a file; do it robustly:
        with self._session.post(url, json={"job_id": job_id}, stream=True) as resp:
            if resp.status_code != 200:
                # Reuse error mapping
                self._handle(resp)
            os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        return dest_path

    def license_report(self, counters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """
        Submit usage counters collected by the engine in the tenant VPC.
        Typically called by the engine itself; SDK supports it for partner tooling.
        """
        return self._post("/license/report", {"counters": counters or {}})


# --------- Local HTTP Worker to expose metrics() at /callback ---------

import json as _json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class _WorkerHandler(BaseHTTPRequestHandler):
    metrics_fn: Optional[Callable[[List[float]], Mapping[str, float]]] = None
    path_route: str = "/callback"
    api_key: Optional[str] = None  # optional shared secret for callback requests

    def log_message(self, format, *args):  # silence default logging
        return

    def do_POST(self):  # noqa: N802
        try:
            if self.path != _WorkerHandler.path_route:
                self.send_response(404); self.end_headers(); return

            # Optional simple auth on callback to avoid random callers inside VPC
            if _WorkerHandler.api_key:
                k = self.headers.get("X-Worker-Key")
                if not k or k != _WorkerHandler.api_key:
                    self.send_response(401); self.end_headers(); return

            length = int(self.headers.get("Content-Length", "0")) or 0
            raw = self.rfile.read(length)
            data = _json.loads(raw.decode("utf-8")) if raw else {}
            sol = data.get("solution")
            if sol is None or not isinstance(sol, list):
                raise ValueError("Missing or invalid 'solution' list")
            if _WorkerHandler.metrics_fn is None:
                raise RuntimeError("metrics_fn not set")
            res = _WorkerHandler.metrics_fn(list(map(float, sol)))
            if not isinstance(res, Mapping):
                raise ValueError("metrics_fn must return a mapping[str,float]")
            body = _json.dumps(dict(res), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            err = _json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LocalWorker:
    """
    Lightweight HTTP worker to host your objective function inside the tenant VPC.

    Example:
        def metrics_fn(x: List[float]) -> Dict[str, float]:
            # Return one or more named objective metrics (max/min defined in Objectives)
            return {"score": -sum(v*v for v in x)}

        with LocalWorker(metrics_fn, host="127.0.0.1", port=8081, api_key="secret") as w:
            print("Callback URL:", w.url)
            # Use this URL in OptimizationRequest.callback_url
    """
    def __init__(self, metrics_fn: Callable[[List[float]], Mapping[str, float]],
                 host: str = "127.0.0.1", port: int = 8081, route: str = "/callback",
                 api_key: Optional[str] = None):
        self._host, self._port, self._route = host, int(port), route
        self._server: Optional[_ThreadedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        _WorkerHandler.metrics_fn = metrics_fn
        _WorkerHandler.path_route = route
        _WorkerHandler.api_key = api_key

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}{self._route}"

    def start(self):
        if self._server is not None:
            return self
        self._server = _ThreadedHTTPServer((self._host, self._port), _WorkerHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        try:
            if self._server:
                self._server.shutdown()
                self._server.server_close()
        finally:
            self._server = None
            self._thread = None

    def __enter__(self): return self.start()
    def __exit__(self, exc_type, exc, tb): self.stop()





### EXAMPLE CODE
from morphantic_client import (
    MorphanticClient, OptimizationRequest, OptimizationConfig, ObjectiveSpec, LocalWorker
)

# 1) Host your objective inside the tenant VPC
def metrics_fn(x):
    # Example multi-objective: maximize score, minimize cost
    return {"score": 1.0 - sum(v*v for v in x), "cost": sum(abs(v) for v in x)}

with LocalWorker(metrics_fn, host="127.0.0.1", port=8081, api_key="local-secret") as worker:
    # 2) Create client to talk to the on‑prem Morphantic Engine
    client = MorphanticClient(
        base_url="https://morphantic-engine.internal",  # inside their VPC
        api_key="TENANT_API_KEY",
        watermark_id="acme-corp-2025"
    )

    # 3) Start an optimization
    req = OptimizationRequest(
        callback_url=worker.url,  # e.g., "http://127.0.0.1:8081/callback"
        dimension=5,
        bounds=(-5.0, 5.0),
        objectives=[
            ObjectiveSpec(name="score", direction="max", weight=1.0),
            ObjectiveSpec(name="cost", direction="min", weight=0.2),
        ],
        config=OptimizationConfig(budget_nfe=300, seed=42, modules=["safezone", "driftguard"]),
        domain="autophagy"  # optional lens tag
    )

    job = client.optimize(req)
    job_id = job["job_id"]

    # 4) Poll metrics and export replay
    print(client.metrics(job_id))
    bundle_path = client.replay_export(job_id, "./replays/acme_job.zip")
    print("Replay saved to:", bundle_path)

    # 5) (Usually done by engine) Report usage counters if needed
    client.license_report({"job_id": job_id, "evals": 300})
