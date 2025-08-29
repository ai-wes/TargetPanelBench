import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from dotenv import load_dotenv, find_dotenv

_client: Optional[MongoClient] = None
_db = None


def init() -> None:
    """Initialize MongoDB client and collections.

    Reads env vars:
    - MONGODB_URI (preferred) or MONGO_URI (legacy)
    - MONGODB_DB or MONGO_DB (default: morphantic)
    """
    global _client, _db
    # Load .env from project root if present
    try:
        load_dotenv(find_dotenv())
    except Exception:
        pass
    uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    if not uri:
        return
    db_name = os.getenv("MONGODB_DB") or os.getenv("MONGO_DB", "morphantic")
    try:
        _client = MongoClient(uri, server_api=ServerApi("1"))
        _db = _client[db_name]
    except Exception:
        _client = None
        _db = None
        return
    # Indexes
    _db.users.create_index([("email", ASCENDING)], unique=True)
    _db.api_keys.create_index([("key_hash", ASCENDING)], unique=True)
    _db.api_keys.create_index([("user_id", ASCENDING)])
    _db.usage_logs.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])


def enabled() -> bool:
    return _db is not None


# ---------------- Users ----------------

def create_user(email: str, first_name: str, last_name: str, hashed_password: str) -> Dict[str, Any]:
    user_id = str(uuid.uuid4())
    doc = {
        "_id": user_id,
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
        "hashed_password": hashed_password,
        "is_active": True,
        "is_verified": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    _db.users.insert_one(doc)
    return to_user_response(doc)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    u = _db.users.find_one({"email": email})
    return u


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    u = _db.users.find_one({"_id": str(user_id)})
    return u


def update_user_names(user_id: str, first_name: Optional[str], last_name: Optional[str]) -> Dict[str, Any]:
    upd = {"updated_at": datetime.utcnow()}
    if first_name:
        upd["first_name"] = first_name
    if last_name:
        upd["last_name"] = last_name
    _db.users.update_one({"_id": str(user_id)}, {"$set": upd})
    u = get_user_by_id(user_id)
    return to_user_response(u)

def set_job_cancelled(job_id: str) -> bool:
    """Mark a job as cancelled"""
    try:
        result = db.jobs.update_one(
            {"_id": ObjectId(job_id)},
            {
                "$set": {
                    "status": "cancelled",
                    "completed_at": datetime.utcnow(),
                    "error_message": "Job cancelled by user"
                }
            }
        )
        return result.modified_count > 0
    except Exception:
        return False

def update_user_password(user_id: str, hashed_password: str) -> None:
    _db.users.update_one({"_id": str(user_id)}, {"$set": {"hashed_password": hashed_password, "updated_at": datetime.utcnow()}})


def to_user_response(u: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": u["_id"],
        "email": u["email"],
        "first_name": u.get("first_name"),
        "last_name": u.get("last_name"),
        "is_active": u.get("is_active", True),
        "is_verified": u.get("is_verified", False),
        "created_at": u.get("created_at"),
    }


# ---------------- API Keys ----------------

def create_api_key(user_id: str, key_hash: str, name: Optional[str], permissions: Optional[List[str]], expires_days: Optional[int], rate_limit: Optional[int]) -> Dict[str, Any]:
    kid = str(uuid.uuid4())
    expires_at = None
    if expires_days:
        expires_at = datetime.utcnow() + timedelta(days=int(expires_days))
    doc = {
        "_id": kid,
        "key_hash": key_hash,
        "user_id": str(user_id),
        "name": name,
        "permissions": permissions or ["optimize", "read_results"],
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "last_used_at": None,
        "is_active": True,
        "rate_limit": rate_limit or 100,
    }
    _db.api_keys.insert_one(doc)
    return {
        "id": kid,
        "name": name,
        "permissions": doc["permissions"],
        "created_at": doc["created_at"],
        "expires_at": expires_at,
    }


def list_api_keys(user_id: str) -> List[Dict[str, Any]]:
    out = []
    for k in _db.api_keys.find({"user_id": str(user_id), "is_active": True}):
        out.append({
            "id": k["_id"],
            "name": k.get("name"),
            "permissions": k.get("permissions", []),
            "created_at": k.get("created_at"),
            "expires_at": k.get("expires_at"),
            "last_used_at": k.get("last_used_at"),
            "is_active": k.get("is_active", True),
            "rate_limit": k.get("rate_limit", 100),
        })
    return out


def revoke_api_key(user_id: str, key_id: str) -> bool:
    res = _db.api_keys.update_one({"_id": str(key_id), "user_id": str(user_id)}, {"$set": {"is_active": False}})
    return res.modified_count > 0


def get_api_key_by_hash(key_hash: str) -> Optional[Dict[str, Any]]:
    return _db.api_keys.find_one({"key_hash": key_hash, "is_active": True})


def mark_api_key_used(key_id: str) -> None:
    _db.api_keys.update_one({"_id": str(key_id)}, {"$set": {"last_used_at": datetime.utcnow()}})


# ---------------- Usage Logs ----------------

def log_usage(user_id: str, api_key_id: Optional[str], endpoint: str, method: str, status_code: int, response_time_ms: int, ip_address: Optional[str], user_agent: Optional[str]) -> None:
    _db.usage_logs.insert_one({
        "_id": str(uuid.uuid4()),
        "user_id": str(user_id),
        "api_key_id": str(api_key_id) if api_key_id else None,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "created_at": datetime.utcnow(),
    })


# ---------------- Jobs ----------------

def create_job(user_id: str, domain: Optional[str], config: Dict[str, Any]) -> Dict[str, Any]:
    jid = str(uuid.uuid4())
    doc = {
        "_id": jid,
        "user_id": str(user_id),
        "status": "queued",
        "domain": domain,
        "config": config,
        "result": None,
        "started_at": None,
        "completed_at": None,
        "error_message": None,
        "created_at": datetime.utcnow(),
    }
    _db.optimization_jobs.insert_one(doc)
    return {"id": jid, "status": "queued"}


def set_job_running(job_id: str):
    _db.optimization_jobs.update_one({"_id": str(job_id)}, {"$set": {"status": "running", "started_at": datetime.utcnow()}})


def set_job_completed(job_id: str, result: Dict[str, Any]):
    _db.optimization_jobs.update_one({"_id": str(job_id)}, {"$set": {"status": "completed", "completed_at": datetime.utcnow(), "result": result}})


def set_job_failed(job_id: str, message: str):
    _db.optimization_jobs.update_one({"_id": str(job_id)}, {"$set": {"status": "failed", "completed_at": datetime.utcnow(), "error_message": message}})


def get_job(job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    return _db.optimization_jobs.find_one({"_id": str(job_id), "user_id": str(user_id)})


def list_jobs(user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
    cur = _db.optimization_jobs.find({"user_id": str(user_id)}).sort("created_at", ASCENDING).skip(int(offset)).limit(int(limit))
    out = []
    for j in cur:
        out.append({
            "id": j["_id"],
            "status": j.get("status"),
            "domain": j.get("domain"),
            "started_at": j.get("started_at"),
            "completed_at": j.get("completed_at"),
            "result": j.get("result"),
            "error_message": j.get("error_message"),
        })
    return out


# ---------------- TNaS Datasets & Runs ----------------

def create_tnas_dataset(user_id: str, name: str, smiles_col: str, label_col: str, source: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    did = str(uuid.uuid4())
    doc = {
        "_id": did,
        "user_id": str(user_id),
        "name": name,
        "smiles_col": smiles_col,
        "label_col": label_col,
        "source": source,
        **meta,
        "created_at": datetime.utcnow(),
    }
    _db.tnas_datasets.insert_one(doc)
    return {
        "dataset_id": did,
        "sha256": doc.get("sha256"),
        "rows": doc.get("rows"),
        "smiles_col": smiles_col,
        "label_col": label_col,
        "created_at": doc["created_at"],
    }

def get_tnas_dataset(user_id: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    return _db.tnas_datasets.find_one({"_id": str(dataset_id), "user_id": str(user_id)})

def create_tnas_run(user_id: str, dataset_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    doc = {
        "_id": rid,
        "user_id": str(user_id),
        "dataset_id": str(dataset_id),
        "status": "queued",
        "config": config,
        "artifacts": {},
        "result": None,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "error_message": None,
    }
    _db.tnas_runs.insert_one(doc)
    return {"run_id": rid, "status": "queued"}

def set_tnas_run_completed(run_id: str, result: Dict[str, Any], artifacts: Dict[str, Any]):
    _db.tnas_runs.update_one(
        {"_id": str(run_id)},
        {"$set": {"status": "completed", "result": result, "artifacts": artifacts, "completed_at": datetime.utcnow()}},
    )

def set_tnas_run_failed(run_id: str, message: str):
    _db.tnas_runs.update_one({"_id": str(run_id)}, {"$set": {"status": "failed", "error_message": message, "completed_at": datetime.utcnow()}})

def get_tnas_run(user_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    return _db.tnas_runs.find_one({"_id": str(run_id), "user_id": str(user_id)})
