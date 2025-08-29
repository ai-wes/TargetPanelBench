import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from mongo_store import enabled as mongo_enabled, get_user_by_id as mongo_get_user_by_id, \
    get_api_key_by_hash as mongo_get_api_key_by_hash, mark_api_key_used as mongo_mark_api_key_used
import re

SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Prefer bcrypt if backend is available; otherwise fall back to pbkdf2_sha256
# This avoids runtime errors on environments without the bcrypt wheels installed.
try:  # pragma: no cover - environment dependent
    import bcrypt as _bcrypt  # noqa: F401
    _HAS_BCRYPT = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_BCRYPT = False

if _HAS_BCRYPT:
    pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")
else:
    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def validate_password_strength(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, expected_type: str = "access") -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {expected_type}",
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Any:
    token = credentials.credentials
    payload = verify_token(token, "access")
    user_id = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
    
    u = mongo_get_user_by_id(user_id)
    # Create a lightweight object with attributes similar to the ORM model
    user = type("_U", (), {**(u or {}), "id": (u or {}).get("_id")}) if u else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    
    return user

def generate_api_key() -> str:
    return f"mpa_{secrets.token_urlsafe(32)}"

def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()

async def validate_api_key(
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[Any]:
    if not api_key:
        return None
    
    key_hash = hash_api_key(api_key)
    doc = mongo_get_api_key_by_hash(key_hash)
    if not doc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired API key")
    if doc.get("expires_at") and doc["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key has expired")
    mongo_mark_api_key_used(doc["_id"])
    # Attach a lightweight object with same attributes expected downstream
    api_key_obj = type("_K", (), {
        "id": doc["_id"],
        "user_id": doc["user_id"],
        "permissions": doc.get("permissions", []),
    })
    return api_key_obj

async def get_current_user_or_api_key(
    api_key: Optional[Any] = Depends(validate_api_key),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    # Prefer API Key if present
    if api_key is not None:
        # Resolve the user associated with this API key
        udoc = mongo_get_user_by_id(api_key.user_id)
        user = type("_U", (), {**(udoc or {}), "id": (udoc or {}).get("_id")}) if udoc else None
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found for API key")
        if not getattr(user, "is_active", True):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
        return {"type": "api_key", "api_key": api_key, "user": user}

    # Fallback to JWT if Authorization bearer is provided
    if credentials and credentials.credentials:
        user = await get_current_user(credentials)
        return {"type": "jwt", "user": user}

    # Neither API key nor JWT provided
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
