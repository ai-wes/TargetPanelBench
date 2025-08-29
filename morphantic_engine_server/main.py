"""
Morphantic Core API - Enhanced with User Authentication and Database Storage
Run with: uvicorn main:app --reload
"""

from datetime import datetime, timedelta
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

import numpy as np
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
 

import os
from dotenv import load_dotenv, find_dotenv
import asyncio
import json as _json
import threading
import asyncio
import json as _json
from mongo_store import init as mongo_init, enabled as mongo_enabled, \
    create_user as mongo_create_user, get_user_by_email as mongo_get_user_by_email, \
    update_user_names as mongo_update_user_names, create_api_key as mongo_create_api_key, \
    list_api_keys as mongo_list_api_keys, revoke_api_key as mongo_revoke_api_key, \
    create_job as mongo_create_job, set_job_running as mongo_set_job_running, \
    set_job_completed as mongo_set_job_completed, set_job_failed as mongo_set_job_failed, \
    get_job as mongo_get_job, list_jobs as mongo_list_jobs, \
    create_tnas_dataset as mongo_create_tnas_dataset, get_tnas_dataset as mongo_get_tnas_dataset, \
    create_tnas_run as mongo_create_tnas_run, set_tnas_run_completed as mongo_set_tnas_run_completed, \
    set_tnas_run_failed as mongo_set_tnas_run_failed, get_tnas_run as mongo_get_tnas_run
from auth import (
    get_password_hash, verify_password, create_access_token, create_refresh_token,
    verify_token, get_current_user, generate_api_key, hash_api_key,
    validate_api_key, get_current_user_or_api_key, validate_password_strength
)
from schemas import (
    UserSignup, UserLogin, UserResponse, TokenResponse, RefreshTokenRequest,
    CreateAPIKeyRequest, APIKeyResponse, APIKeyInfo, UsageStats,
    PasswordResetRequest, PasswordResetConfirm, ChangePasswordRequest,
    UpdateUserProfile, NarrowingRequest, NarrowingWeights
)
from pydantic import BaseModel, Field, validator
from enum import Enum

from complete_teai_methods_slim_v2 import AdvancedArchipelagoEvolution, ObjectiveSpec
from morphantic_modules import attach_modules

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load environment from .env (search up the tree)
    try:
        load_dotenv(find_dotenv())
    except Exception:
        pass
    # Initialize MongoDB (required)
    mongo_init()
    yield

app = FastAPI(
    title="Morphantic Core API",
    description="Advanced optimization API with user authentication and API key management",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static frontend (login/signup + gated app) ----
try:
    if os.path.isdir("web"):
        app.mount("/app", StaticFiles(directory="web", html=True), name="app")
except Exception:
    pass

@app.get("/", include_in_schema=False)
async def root_index():
    if os.path.isdir("web"):
        return RedirectResponse(url="/app/login.html")
    return {"status": "ok"}
 

# ---- WebSocket eval broker (optional) ----
# Clients connect to /v1/eval/ws with X-API-Key header.
# Server will push eval tasks: {type:"eval", task_id, solution, timeout_s}
# Clients respond: {type:"result", task_id, metrics} or {type:"error", task_id, message}

class _WSClient:
    def __init__(self, ws: WebSocket, user_id: str):
        self.ws = ws
        self.user_id = user_id
        self.pending: Dict[str, asyncio.Future] = {}

_WS_CLIENTS: Dict[str, _WSClient] = {}

@app.websocket("/v1/eval/ws")
async def eval_ws(websocket: WebSocket):
    # Manual API key auth for WS
    api_key = websocket.headers.get("X-API-Key")
    if not api_key:
        await websocket.close(code=4401)
        return
    try:
        from auth import validate_api_key as _validate, mongo_get_user_by_id  # type: ignore
        api_key_obj = await _validate(api_key)
        if api_key_obj is None:
            await websocket.close(code=4403)
            return
        udoc = mongo_get_user_by_id(api_key_obj.user_id)
        user_id = str(udoc.get("_id")) if udoc else None
        if not user_id:
            await websocket.close(code=4401)
            return
    except Exception:
        await websocket.close(code=4403)
        return

    await websocket.accept()
    client = _WSClient(websocket, user_id)
    _WS_CLIENTS[user_id] = client
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = _json.loads(msg)
                t = data.get("type")
                if t in ("result", "error"):
                    tid = data.get("task_id")
                    fut = client.pending.pop(tid, None)
                    if fut and not fut.done():
                        if t == "result":
                            fut.set_result({"ok": True, "metrics": data.get("metrics", {})})
                        else:
                            fut.set_result({"ok": False, "message": data.get("message", "error")})
                elif t == "ping":
                    await websocket.send_text(_json.dumps({"type": "pong"}))
            except Exception:
                # ignore malformed messages
                pass
    except WebSocketDisconnect:
        pass
    finally:
        for fut in list(client.pending.values()):
            if not fut.done():
                fut.set_result({"ok": False, "message": "client disconnected"})
        _WS_CLIENTS.pop(user_id, None)

async def _eval_via_ws(user_id: str, x: List[float], timeout_s: int = 60) -> Dict[str, Any]:
    client = _WS_CLIENTS.get(str(user_id))
    if not client:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="WS client not connected")
    task_id = str(uuid.uuid4())
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    client.pending[task_id] = fut
    try:
        await client.ws.send_text(_json.dumps({
            "type": "eval",
            "task_id": task_id,
            "solution": list(map(float, x)),
            "timeout_s": int(timeout_s),
        }))
    except Exception:
        client.pending.pop(task_id, None)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="WS send failed")
    try:
        res = await asyncio.wait_for(fut, timeout=float(timeout_s))
        if res.get("ok"):
            return res.get("metrics", {})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=res.get("message", "eval error"))
    finally:
        client.pending.pop(task_id, None)

@app.post("/v1/auth/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup):
    """Register a new user account"""
    
    if mongo_get_user_by_email(user_data.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    is_valid, message = validate_password_strength(user_data.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return mongo_create_user(
        email=user_data.email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        hashed_password=get_password_hash(user_data.password),
    )

@app.post("/v1/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """Authenticate user and receive JWT tokens"""
    
    user_doc = mongo_get_user_by_email(credentials.email)
    user = type("_U", (), {**user_doc, "id": user_doc["_id"]}) if user_doc else None
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800
    )

@app.post("/v1/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    
    payload = verify_token(request.refresh_token, "refresh")
    user_id = payload.get("sub")
    
    from mongo_store import get_user_by_id as _get
    user = _get(user_id)
    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    access_token = create_access_token(data={"sub": str(user.get('_id'))})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=request.refresh_token,
        expires_in=1800
    )

@app.get("/v1/auth/me", response_model=UserResponse)
async def get_current_user_profile(current_user = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@app.put("/v1/auth/me", response_model=UserResponse)
async def update_user_profile(
    update_data: UpdateUserProfile,
    current_user = Depends(get_current_user),
):
    """Update current user profile"""
    
    return mongo_update_user_names(str(current_user.id), update_data.first_name, update_data.last_name)

@app.post("/v1/auth/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user = Depends(get_current_user),
):
    """Change user password"""
    
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    is_valid, message = validate_password_strength(request.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    from auth import get_password_hash as _gph
    from mongo_store import update_user_password as _upd
    _upd(str(current_user.id), _gph(request.new_password))
    return {"message": "Password changed successfully"}

@app.post("/v1/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user = Depends(get_current_user),
):
    """Create a new API key for the authenticated user.

    Requires JWT auth and confirmation of the user's current password.
    """

    # Confirm password for key creation
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Current password is incorrect"
        )

    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)

    meta = mongo_create_api_key(
        user_id=str(current_user.id),
        key_hash=key_hash,
        name=request.name,
        permissions=request.permissions,
        expires_days=request.expires_days,
        rate_limit=request.rate_limit,
    )
    return APIKeyResponse(
        api_key=api_key,
        id=meta["id"],
        name=meta.get("name"),
        expires_at=meta.get("expires_at"),
        permissions=meta.get("permissions", []),
        created_at=meta.get("created_at"),
    )

@app.get("/v1/api-keys", response_model=List[APIKeyInfo])
async def list_api_keys(
    current_user = Depends(get_current_user),
):
    """List all API keys for the authenticated user"""
    
    return mongo_list_api_keys(str(current_user.id))

@app.delete("/v1/api-keys/{key_id}")
async def revoke_api_key(
    key_id: uuid.UUID,
    current_user = Depends(get_current_user),
):
    """Revoke a specific API key"""
    
    ok = mongo_revoke_api_key(str(current_user.id), str(key_id))
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    return {"message": "API key revoked successfully"}

@app.get("/v1/usage/stats", response_model=UsageStats)
async def get_usage_stats(
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
):
    """Get usage statistics for the authenticated user"""
    
    user = auth["user"]
    now = datetime.utcnow()
    
    # TODO: wire logging middleware to populate usage_logs in Mongo; return zeros for now
    return UsageStats(
        total_requests=0,
        successful_requests=0,
        failed_requests=0,
        average_response_time_ms=0.0,
        requests_last_24h=0,
        requests_last_7d=0,
        requests_last_30d=0,
    )

# Models and routes below

# ---- Models ----
class OptimizerConfig(BaseModel):
    pop_size: int = Field(60, gt=0, le=500)
    max_generations: int = Field(45, gt=0, le=1000)
    n_islands: int = Field(4, gt=0, le=16)
    seed: Optional[int] = None

class ConstraintExpr(BaseModel):
    name: str
    op: str
    value: float
    penalty: Optional[float] = None

class Constraints(BaseModel):
    hard: Optional[List[ConstraintExpr]] = None
    soft: Optional[List[ConstraintExpr]] = None

class OptimizationRequest(BaseModel):
    objectives: Optional[List[ObjectiveSpec]] = Field(None, description="Custom objectives")
    callback_url: str
    dimension: int = Field(..., gt=0, le=2048)
    bounds: Tuple[float, float]
    config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    modules: Optional[List[str]] = Field(None, description="List of module specs")
    domain: Optional[str] = Field(None, description="Specialty domain")
    mode: Optional[str] = Field("so", description='Optimization mode: "so" or "mo"')
    constraints: Optional[Constraints] = None
    use_websocket: Optional[bool] = Field(False, description="Use WebSocket eval if connected")

    @validator('objectives', always=True)
    def check_objectives_or_domain(cls, v, values):
        if not values.get('domain') and not v:
            raise ValueError("Either 'domain' or 'objectives' must be provided")
        return v

class BatchEvaluateRequest(BaseModel):
    callback_url: str
    X: List[List[float]]
    timeout_s: Optional[int] = Field(60, gt=0, le=600)
    use_websocket: Optional[bool] = Field(False, description="Use WebSocket eval if connected")

class BatchEvaluateResponse(BaseModel):
    results: List[Dict[str, Any]]

class TNaSDatasetIn(BaseModel):
    name: str
    smiles_col: str
    label_col: str
    source: str = Field("path", description="upload|url|path")
    url: Optional[str] = None
    path: Optional[str] = None

class TNaSDatasetOut(BaseModel):
    dataset_id: str
    sha256: Optional[str] = None
    rows: Optional[int] = None
    smiles_col: str
    label_col: str

class TNaSRunIn(BaseModel):
    dataset_id: str
    k: int = Field(ge=1, default=12)
    seed: int = 123
    test_frac: float = Field(0.2, gt=0.0, lt=0.9)
    strategy: str = Field("morphantic", description='morphantic|optuna|optuna_eom|nsga2')
    mode: str = Field("scalar", description='scalar|mo')
    weights: Optional[Dict[str, float]] = None
    constraints: Optional[Constraints] = None
    use_websocket: Optional[bool] = Field(False, description="Use WebSocket eval if connected")
    modules: Optional[List[str]] = None
    budget: int = Field(400, ge=1)

class TNaSRunOut(BaseModel):
    run_id: str
    metrics_eval: Dict[str, Any]
    metrics_model: Dict[str, Any]
    selection: Dict[str, Any]
    artifacts: Dict[str, Any]
    elapsed_s: float

class TNaSAbIn(BaseModel):
    dataset_id: str
    k: int = Field(ge=1, default=12)
    seed: int = 123
    test_frac: float = Field(0.2, gt=0.0, lt=0.9)
    budget: int = Field(400, ge=1)
    lanes: List[str] = Field(default_factory=lambda: ["morphantic", "optuna", "optuna_eom", "nsga2"])

class TNaSAbOut(BaseModel):
    leaderboard: List[Dict[str, Any]]
    artifacts: Dict[str, Any]

class TNaSFeedbackIn(BaseModel):
    run_id: str
    labels: List[Dict[str, int]]

class TNaSFeedbackOut(BaseModel):
    run_id: str
    selection: Dict[str, Any]
    metrics_model: Dict[str, Any]
    metrics_eval: Dict[str, Any]
    artifacts: Dict[str, Any]

class TNaSShortlistIn(BaseModel):
    dataset_id: str
    k: int = Field(ge=1, default=20)
    budget: int = Field(ge=1, default=120)
    explore_bias: float = Field(0.5, ge=0.0, le=1.0)

class TNaSShortlistOut(BaseModel):
    selection: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]

# ---- Auth helpers ----
def _has_permission(auth: Dict[str, Any], perm: str) -> bool:
    try:
        if not auth or auth.get("type") != "api_key":
            # Non-API-key flows (JWT) are handled elsewhere; treat as permitted here
            return True
        ak = auth.get("api_key")
        if ak is None:
            return False
        if isinstance(ak, dict):
            perms = ak.get("permissions", [])
        else:
            perms = getattr(ak, "permissions", [])
        return perm in (perms or [])
    except Exception:
        return False

# Dedicated loop for WS sync calls from background tasks
_WS_LOOP = None
_WS_THREAD = None

def _ensure_ws_loop():
    global _WS_LOOP, _WS_THREAD
    if _WS_LOOP is None:
        loop = asyncio.new_event_loop()
        def run():
            asyncio.set_event_loop(loop)
            loop.run_forever()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        _WS_LOOP = loop
        _WS_THREAD = t

def _eval_via_ws_sync(user_id: str, x: list[float], timeout_s: int = 60) -> dict:
    _ensure_ws_loop()
    fut = asyncio.run_coroutine_threadsafe(_eval_via_ws(user_id, x, timeout_s=timeout_s), _WS_LOOP)
    res = fut.result(timeout=timeout_s + 5)
    return res

# ---- Routes: batch evaluate ----
@app.post("/v1/batch_evaluate", response_model=BatchEvaluateResponse)
async def batch_evaluate(
    request: BatchEvaluateRequest,
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
):
    if not _has_permission(auth, "optimize"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key lacks optimize permission")
    out: List[Dict[str, Any]] = []
    timeout = int(request.timeout_s or 60)
    user = auth["user"]
    for x in request.X:
        try:
            if bool(request.use_websocket):
                metrics = await _eval_via_ws(str(user.id), x, timeout_s=timeout)
                out.append(metrics)
            else:
                resp = requests.post(request.callback_url, json={"solution": list(map(float, x))}, timeout=timeout)
                resp.raise_for_status()
                out.append(resp.json())
        except Exception as e:
            out.append({"error": str(e)})
    return BatchEvaluateResponse(results=out)

# ---- TNaS helpers and routes ----
def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _tnas_diversity(smiles_list: List[str]) -> float:
    if not smiles_list or len(smiles_list) < 2:
        return 0.0
    sims = []
    sets = [set(s) for s in smiles_list]
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            a,b = sets[i], sets[j]
            inter=len(a & b); union=max(1,len(a|b))
            sims.append(inter/union)
    return float(1.0 - (sum(sims)/len(sims))) if sims else 0.0

def _eval_constraints_for_selection(df, sel_idx: List[int], constraints: Optional[Constraints]):
    viol_counts = {}
    total_penalty = 0.0
    hard_violated = False
    def agg_value(name: str) -> Optional[float]:
        if name in df.columns and len(sel_idx) > 0:
            try:
                vals = df.loc[sel_idx, name].astype(float)
                return float(vals.mean())
            except Exception:
                return None
        return None
    def violation(val: float, op: str, ref: float) -> float:
        if op == "<=": return max(0.0, val - ref)
        if op == "<": return max(0.0, val - ref + 1e-12)
        if op == ">=": return max(0.0, ref - val)
        if op == ">": return max(0.0, ref - val + 1e-12)
        if op == "==": return 0.0 if abs(val - ref) <= 1e-9 else abs(val - ref)
        if op == "!=": return 0.0 if abs(val - ref) > 1e-9 else 1.0
        return 0.0
    if not constraints:
        return total_penalty, hard_violated, viol_counts
    if constraints.hard:
        for c in constraints.hard:
            v = agg_value(c.name)
            if v is None: continue
            if violation(v, c.op, float(c.value)) > 0:
                hard_violated = True
                viol_counts[c.name] = viol_counts.get(c.name, 0) + 1
    if constraints.soft:
        for c in constraints.soft:
            v = agg_value(c.name)
            if v is None: continue
            viol = violation(v, c.op, float(c.value))
            if viol > 0:
                total_penalty += float(c.penalty or 1.0) * viol
                viol_counts[c.name] = viol_counts.get(c.name, 0) + 1
    return total_penalty, hard_violated, viol_counts

@app.post("/tnas/datasets", response_model=TNaSDatasetOut)
async def tnas_create_dataset(body: TNaSDatasetIn, auth: Dict[str, Any] = Depends(get_current_user_or_api_key)):
    user = auth["user"]
    rows = sha = None
    if body.source == "path" and body.path:
        import pandas as _pd
        try:
            df = _pd.read_csv(body.path)
            rows = int(df.shape[0])
            sha = _sha256_file(body.path)
        except Exception:
            pass
    meta = {"rows": rows, "sha256": sha, "path": body.path, "url": body.url}
    return mongo_create_tnas_dataset(str(user.id), body.name, body.smiles_col, body.label_col, body.source, meta)

def _run_tnas_background(run_id: str, user_id: str, cfg: TNaSRunIn):
    import time as _time
    import json as _json
    from pathlib import Path as _Path
    import pandas as _pd
    ds = mongo_get_tnas_dataset(user_id, cfg.dataset_id)
    if not ds:
        mongo_set_tnas_run_failed(run_id, "Dataset not found"); return
    csv_path = ds.get("path")
    if not csv_path:
        mongo_set_tnas_run_failed(run_id, "Dataset path unavailable"); return
    df = _pd.read_csv(csv_path).reset_index(drop=True)
    label_col = ds.get("label_col")
    rng = np.random.default_rng(int(cfg.seed))
    n, k = int(len(df)), int(cfg.k)
    label_vals = None
    if label_col in df.columns:
        try:
            label_vals = df[label_col].astype(float).to_numpy()
        except Exception:
            label_vals = None
    max_samples = max(200, min(2000, int(cfg.budget) * 3))
    t0 = _time.perf_counter()
    best = None; best_score = -1e9
    portfolios = []
    for _ in range(max_samples):
        idxs = list(range(n)) if n <= k else rng.choice(n, size=k, replace=False).tolist()
        hits = float(np.sum(label_vals[idxs]))/max(1,k) if label_vals is not None else 0.0
        smiles_list = df.loc[idxs, 'smiles'].astype(str).tolist() if 'smiles' in df.columns else [str(i) for i in idxs]
        div = _tnas_diversity(smiles_list)
        pen, hard_viol, vcounts = _eval_constraints_for_selection(df, idxs, cfg.constraints)
        portfolios.append((idxs, hits, div, pen, hard_viol, vcounts))
        if not hard_viol:
            s = float((cfg.weights or {}).get('activity',0.7))*hits + float((cfg.weights or {}).get('diversity',0.3))*div - pen
            if s > best_score:
                best_score, best = s, (idxs, hits, div, pen)
    if best is None:
        idxs = list(range(min(k,n)))
        hits = float(np.sum(label_vals[idxs]))/max(1,k) if label_vals is not None else 0.0
        div = _tnas_diversity(df.loc[idxs,'smiles'].astype(str).tolist()) if 'smiles' in df.columns else 0.0
        pen = 0.0
    else:
        idxs, hits, div, pen = best
    sel_smiles = df.loc[idxs, 'smiles'].astype(str).tolist() if 'smiles' in df.columns else [str(i) for i in idxs]
    sel = {"indices": idxs, "smiles": sel_smiles}
    metrics_model = {"hit_rate": float(hits), "diversity": float(div), "penalty": float(pen)}
    metrics_eval = {"k": k, "N_pool": int(n), "hits_in_selection": int(round(hits*k)) if label_vals is not None else None, "hit_rate": float(hits)}
    # PF and hv2d (optional)
    front = []; hv2d = None
    if (cfg.mode or 'scalar').lower() == 'mo':
        pts = [((h,d), idxs) for idxs,h,d,pen,hard,_ in portfolios if not hard]
        nd = []
        for (a1,a2), idl in pts:
            if not any((b1>=a1 and b2>=a2) and (b1>a1 or b2>a2) for (b1,b2),_ in pts):
                nd.append(((a1,a2), idl))
        nd = nd[:min(50,len(nd))]
        front = [{"x": idl, "F": [float(a1), float(a2)]} for (a1,a2), idl in nd]
        if nd:
            pts_sorted = sorted([p for p,_ in nd], key=lambda t: t[0])
            area=0.0; prev_x=0.0
            for x,y in pts_sorted:
                dx=max(0.0, x-prev_x); area += dx*max(0.0,y); prev_x=x
            hv2d=float(area)
    # artifacts
    art_root = _Path(os.getenv("TNAS_STORE_DIR", "artifacts/tnas")); art_dir = art_root / run_id; art_dir.mkdir(parents=True, exist_ok=True)
    with open(art_dir/"run.json", 'w', encoding='utf-8') as f:
        _json.dump({"run_id":run_id,"dataset_id":cfg.dataset_id,"selection":sel,"metrics_eval":metrics_eval,"metrics_model":metrics_model}, f, indent=2)
    artifacts = {"dir": str(art_dir), "run.json": str(art_dir/"run.json")}
    payload = {"metrics_eval": metrics_eval, "metrics_model": metrics_model, "selection": sel, "elapsed_s": float(_time.perf_counter()-t0)}
    if (cfg.mode or 'scalar').lower() == 'mo': payload.update({"front": front, "hv2d": hv2d})
    mongo_set_tnas_run_completed(run_id, payload, artifacts)

@app.post("/tnas/runs", response_model=TNaSRunOut)
async def tnas_start_run(body: TNaSRunIn, background_tasks: BackgroundTasks, auth: Dict[str, Any] = Depends(get_current_user_or_api_key)):
    user = auth["user"]
    run = mongo_create_tnas_run(str(user.id), body.dataset_id, body.dict())
    background_tasks.add_task(_run_tnas_background, run["run_id"], str(user.id), body)
    j = mongo_get_tnas_run(str(user.id), run["run_id"]) or {}
    return TNaSRunOut(run_id=run["run_id"], metrics_eval=j.get("result", {}).get("metrics_eval", {}), metrics_model=j.get("result", {}).get("metrics_model", {}), selection=j.get("result", {}).get("selection", {}), artifacts=j.get("artifacts", {}), elapsed_s=j.get("result", {}).get("elapsed_s", 0.0))

@app.get("/tnas/runs/{run_id}")
async def tnas_get_run(run_id: str, auth: Dict[str, Any] = Depends(get_current_user_or_api_key)):
    user = auth["user"]
    r = mongo_get_tnas_run(str(user.id), str(run_id))
    if not r: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return {"run_id": r.get("_id"), "status": r.get("status"), "result": r.get("result"), "artifacts": r.get("artifacts"), "created_at": r.get("created_at"), "completed_at": r.get("completed_at"), "error_message": r.get("error_message")}

@app.get("/tnas/runs/{run_id}/artifacts/{name}")
async def tnas_get_artifact(run_id: str, name: str, auth: Dict[str, Any] = Depends(get_current_user_or_api_key)):
    user = auth["user"]
    r = mongo_get_tnas_run(str(user.id), str(run_id))
    if not r: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    path = (r.get("artifacts", {}) or {}).get(name)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return FileResponse(path)


@app.post("/tnas/ab", response_model=TNaSAbOut)
async def tnas_ab(body: TNaSAbIn, auth: Dict[str, Any] = Depends(get_current_user_or_api_key)):
    user = auth["user"]
    ds = mongo_get_tnas_dataset(str(user.id), body.dataset_id)
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    csv_path = ds.get("path")
    if not csv_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset path not available")
    import pandas as _pd
    import json as _json
    from pathlib import Path as _Path
    df = _pd.read_csv(csv_path).reset_index(drop=True)
    label_col = ds.get("label_col")
    rng = np.random.default_rng(int(body.seed))

    n = int(len(df)); k = int(body.k)
    label_vals = None
    if label_col in df.columns:
        try:
            label_vals = df[label_col].astype(float).to_numpy()
        except Exception:
            label_vals = None
    max_samples = max(200, min(2000, int(body.budget) * 3))

    leaderboard: List[Dict[str, Any]] = []
    for lane in body.lanes:
        best = None; best_score = -1e9
        for _ in range(max_samples):
            idxs = list(range(n)) if n <= k else rng.choice(n, size=k, replace=False).tolist()
            hits = float(np.sum(label_vals[idxs]))/max(1,k) if label_vals is not None else 0.0
            smiles_list = df.loc[idxs, 'smiles'].astype(str).tolist() if 'smiles' in df.columns else [str(i) for i in idxs]
            div = _tnas_diversity(smiles_list)
            score = 0.7 * hits + 0.3 * div
            if score > best_score:
                best_score = score
                best = (idxs, hits, div)
        idxs, hits, div = best
        leaderboard.append({
            "lane": lane,
            "score": float(best_score),
            "metrics": {"hit_rate": float(hits), "diversity": float(div)},
            "selection": {"indices": idxs},
        })

    out_root = _Path(os.getenv("TNAS_AB_STORE_DIR", "artifacts/tnas_ab"))
    out_root.mkdir(parents=True, exist_ok=True)
    cmp_path = out_root / f"ab_{body.dataset_id}.json"
    cmp_path.write_text(_json.dumps({"leaderboard": leaderboard}, indent=2), encoding="utf-8")
    return TNaSAbOut(leaderboard=leaderboard, artifacts={"compare.json": str(cmp_path)})



@app.post("/v1/targets/narrowing")
def targets_narrowing(body: NarrowingRequest, auth: Dict[str, Any] = Depends(get_current_user_or_api_key)):
    user = auth["user"]
    # Strategy A (recommended): callback evaluates candidate metrics given a solution
    # Strategy B: if candidates are provided, evaluate them internally and rank directly (no callback).
    try:
        if body.candidates:
            # B: direct ranking path (simple scoring demo). Replace with your real evaluation.
            # Here we synthesize objective scores per candidate; in production, call your scoring stack.
            rng = np.random.default_rng(int(body.seed or 123))
            ranking = []
            for c in body.candidates:
                scores = {
                    "activity": float(rng.random()),
                    "selectivity": float(rng.random()),
                    "novelty": float(rng.random()),
                    "safety": float(rng.random())
                }
                w = body.weights or NarrowingWeights()
                priority = (
                    w.activity * scores["activity"] +
                    w.selectivity * scores["selectivity"] +
                    w.novelty * scores["novelty"] +
                    w.safety * scores["safety"]
                )
                ranking.append({
                    "symbol": c.get("symbol"),
                    "name": c.get("name"),
                    "ext_id": c.get("ext_id"),
                    "scores": scores,
                    "priority_score": float(priority),
                })
            ranking.sort(key=lambda r: r["priority_score"], reverse=True)
            return {
                "job_id": None,
                "status": "completed",
                "result": { "ranking": ranking[:int(body.k)] },
                "completed_at": datetime.utcnow().isoformat() + "Z"
            }

        # A: use your existing optimization stack with AEA + callback for evaluations
        # Build a multi-objective run. The evaluation endpoint should return named metrics:
        # {activity, selectivity, novelty, safety}. AEA minimizes scalarized loss — use modules if you like.
        if not body.callback_url or not body.dimension or not body.bounds:
            raise HTTPException(status_code=400, detail="Provide callback_url, dimension, and bounds for callback-drivennarrowing")

        # Leverage your existing job system
        req = OptimizationRequest(
            callback_url=body.callback_url,
            dimension=int(body.dimension),
            bounds=tuple(body.bounds),
            config=OptimizerConfig(seed=body.seed),
            domain="MorphanticBio",
            mode="mo",
            modules=body.modules or ["diversity"],
            objectives=[
                # encode direction via baseline>target (minimize) or target>baseline (maximize)
                ObjectiveSpec(name="activity", weight= (body.weights or NarrowingWeights()).activity, baseline=0.3,
target=0.9, direction="max"),
                ObjectiveSpec(name="selectivity", weight=(body.weights or NarrowingWeights()).selectivity, baseline=0.3,
target=0.9, direction="max"),
                ObjectiveSpec(name="novelty", weight=(body.weights or NarrowingWeights()).novelty, baseline=0.3,
target=0.9, direction="max"),
                ObjectiveSpec(name="safety", weight=(body.weights or NarrowingWeights()).safety, baseline=0.3,
target=0.9, direction="max"),
            ],
            use_websocket=False
        )
        # Start async job exactly like /v1/optimize
        job = mongo_create_job(user_id=str(user.id), domain="target_narrowing", config=req.config.dict())
        # Run in background
        def _bg():
            try:
                run_optimization_task(job["id"], req, str(user.id))
            except Exception as e:
                mongo_set_job_failed(job["id"], str(e))
        threading.Thread(target=_bg, daemon=True).start()
        return {"job_id": job["id"], "message": "Target narrowing job started", "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def run_optimization_task(
        job_id: str,
        request: OptimizationRequest,
        user_id: str,
    ):
        """Background task for running optimization"""
        
        mongo_set_job_running(job_id)
        
        try:
            final_objectives = request.objectives
            final_modules = request.modules if request.modules is not None else []
            
            if request.domain and not final_objectives:
                if request.domain == "MorphanticBio":
                    final_objectives = [
                        ObjectiveSpec(name="activity", weight=0.5, baseline=0.3, target=0.9, direction="max"),
                        ObjectiveSpec(name="qed_score", weight=0.3, baseline=0.4, target=0.8, direction="max"),
                        ObjectiveSpec(name="synthetic_accessibility", weight=0.2, baseline=8.0, target=3.0, direction="min")
                    ]
                    if not any("diversity" in m for m in final_modules):
                        final_modules.append("diversity")
                
            def metrics_fn_via_callback(x: np.ndarray) -> Dict[str, float]:
                try:
                    if bool(request.use_websocket):
                        return _eval_via_ws_sync(user_id, x.tolist(), timeout_s=60)
                    else:
                        response = requests.post(
                            request.callback_url,
                            json={'solution': x.tolist()},
                            timeout=60
                        )
                        response.raise_for_status()
                        return response.json()
                except Exception:
                    return {obj.name: obj.baseline for obj in final_objectives}
            # Build scalarized fitness with constraints (works for SO and simple MO via weights)
            def _scalarize(metrics: Dict[str, float], objectives: List[ObjectiveSpec]) -> float:
                loss = 0.0
                for obj in objectives:
                    mval = float(metrics.get(obj.name, obj.baseline))
                    # Normalize toward target vs baseline
                    if obj.direction == "min":
                        denom = max(1e-9, (obj.baseline - obj.target))
                        l = max(0.0, (mval - obj.target) / denom)
                    else:  # max
                        denom = max(1e-9, (obj.target - obj.baseline))
                        l = max(0.0, (obj.target - mval) / denom)
                    loss += float(obj.weight) * l
                return loss

            def _violation(val: float, op: str, ref: float) -> float:
                if op == "<=":
                    return max(0.0, val - ref)
                if op == "<":
                    return max(0.0, val - ref + 1e-12)
                if op == ">=":
                    return max(0.0, ref - val)
                if op == ">":
                    return max(0.0, ref - val + 1e-12)
                if op == "==":
                    return 0.0 if abs(val - ref) <= 1e-9 else abs(val - ref)
                if op == "!=":
                    return 0.0 if abs(val - ref) > 1e-9 else 1.0
                return 0.0

            def fitness_with_constraints(x: np.ndarray) -> float:
                # CHECK FOR CANCELLATION BEFORE EACH EVALUATION
                current_job = mongo_get_job(job_id, user_id)
                if current_job and current_job.get("status") == "cancelled":
                    raise Exception("Job was cancelled by user")
                
                metrics = metrics_fn_via_callback(x)
                base = _scalarize(metrics, final_objectives)
                penalty = 0.0
                if request.constraints:
                    if request.constraints.hard:
                        hard_viol = 0.0
                        for c in request.constraints.hard:
                            v = float(metrics.get(c.name, 0.0))
                            hard_viol += _violation(v, c.op, float(c.value))
                        if hard_viol > 0:
                            penalty += 1e6 * hard_viol
                    if request.constraints.soft:
                        for c in request.constraints.soft:
                            v = float(metrics.get(c.name, 0.0))
                            viol = _violation(v, c.op, float(c.value))
                            if viol > 0:
                                penalty += float(c.penalty or 1.0) * viol
                return base + penalty

            aea = AdvancedArchipelagoEvolution(
                dimension=request.dimension,
                bounds=request.bounds,
                pop_size=request.config.pop_size,
                max_generations=request.config.max_generations,
                n_islands=request.config.n_islands,
                seed=request.config.seed
            )

            final_fitness_func = attach_modules(
                inner_fitness=fitness_with_constraints,
                dim=request.dimension,
                bounds=request.bounds,
                budget_max=(request.config.pop_size * request.config.max_generations),
                seed=request.config.seed,
                modules=final_modules,
                problem_name=request.domain or "custom"
            )

            result, best_cell = aea.optimize(fitness_func=final_fitness_func)
            
            best_x = best_cell.get_solution().tolist()
            best_metrics = metrics_fn_via_callback(np.asarray(best_x, float))
            payload = {
                "best_solution": best_x,
                "best_fitness": result.final_fitness,
                "final_metrics": best_metrics,
                "mode": request.mode or "so",
            }
            # For MO, return a trivial front containing the pick for now (can expand later)
            if (request.mode or "so").lower() == "mo":
                payload.update({
                    "front": [
                        {"x": best_x, "F": [best_metrics.get(obj.name) for obj in (final_objectives or [])]}
                    ],
                    "pick": {"x": best_x, "F": [best_metrics.get(obj.name) for obj in (final_objectives or [])]},
                })
            mongo_set_job_completed(job_id, payload)
        except Exception as e:
            mongo_set_job_failed(job_id, str(e))

@app.post("/v1/optimize", status_code=status.HTTP_202_ACCEPTED)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
):
    """Start an optimization job"""
    
    user = auth["user"]
    
    if not _has_permission(auth, "optimize"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key lacks optimize permission"
        )
    
    job = mongo_create_job(user_id=str(user.id), domain=request.domain, config=request.config.dict())
    background_tasks.add_task(run_optimization_task, job["id"], request, str(user.id))
    return {"job_id": job["id"], "message": "Optimization job started"}

@app.get("/v1/results/{job_id}")
async def get_job_results(
    job_id: uuid.UUID,
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
):
    """Get optimization job results"""
    
    user = auth["user"]
    
    j = mongo_get_job(str(job_id), str(user.id))
    if not j:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return {
        "job_id": j.get("_id"),
        "status": j.get("status"),
        "domain": j.get("domain"),
        "started_at": j.get("started_at"),
        "completed_at": j.get("completed_at"),
        "result": j.get("result"),
        "error_message": j.get("error_message"),
    }
    
    
    
    
@app.delete("/v1/jobs/{job_id}")
async def cancel_job(
    job_id: uuid.UUID,
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
):
    """Cancel a running optimization job"""
    
    user = auth["user"]
    
    # Get the job to verify ownership
    j = mongo_get_job(str(job_id), str(user.id))
    if not j:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    
    # Check if job can be cancelled
    status = j.get("status", "")
    if status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Cannot cancel job with status: {status}"
        )
    
    # Mark job as cancelled in database
    from mongo_store import set_job_cancelled
    success = set_job_cancelled(str(job_id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job"
        )
    
    return {
        "job_id": str(job_id),
        "message": "Job cancelled successfully",
        "status": "cancelled"
    }
    
    


@app.get("/v1/jobs")
async def list_jobs(
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
    limit: int = 10,
    offset: int = 0
):
    """List optimization jobs for the authenticated user"""
    
    user = auth["user"]
    
    jobs = mongo_list_jobs(str(user.id), limit=limit, offset=offset)
    return {"jobs": jobs}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
