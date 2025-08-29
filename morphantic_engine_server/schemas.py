from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, validator
import uuid

class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    
    @validator('first_name', 'last_name')
    def validate_name(cls, v):
        if not v.replace(" ", "").replace("-", "").isalpha():
            raise ValueError('Name must contain only letters, spaces, and hyphens')
        return v.strip()

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    first_name: str
    last_name: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(default=1800, description="Token expiry in seconds")

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class CreateAPIKeyRequest(BaseModel):
    current_password: str = Field(..., min_length=8, description="Confirm your account password to create a new API key")
    name: Optional[str] = Field(None, max_length=100, description="Optional name for the API key")
    expires_days: Optional[int] = Field(30, ge=1, le=365, description="Days until key expires (null for no expiry)")
    permissions: Optional[List[str]] = Field(
        default=["optimize", "read_results"],
        description="Permissions for the API key"
    )
    rate_limit: Optional[int] = Field(100, ge=1, le=10000, description="Rate limit per hour")

class APIKeyResponse(BaseModel):
    api_key: str
    id: uuid.UUID
    name: Optional[str]
    expires_at: Optional[datetime]
    permissions: List[str]
    created_at: datetime
    message: str = "Store this API key securely - it won't be shown again"

class APIKeyInfo(BaseModel):
    id: uuid.UUID
    name: Optional[str]
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool
    rate_limit: int
    
    class Config:
        from_attributes = True

class UsageStats(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    requests_last_24h: int
    requests_last_7d: int
    requests_last_30d: int

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

class UpdateUserProfile(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)



class NarrowingWeights(BaseModel):
    # knobs; default weights sum does not need to =1; they’re used to scalarize when needed
    activity: float = 0.5
    selectivity: float = 0.2
    novelty: float = 0.15
    safety: float = 0.15

class NarrowingRequest(BaseModel):
    disease: str
    k: int = Field(ge=1, le=100, default=10)
    weights: Optional[NarrowingWeights] = None
    feature_set_version: Optional[str] = None
    # Either pass candidates, or rely on callback to compute on-the-fly
    candidates: Optional[List[Dict[str, Any]]] = None  # e.g., [{symbol,name,ext_id},...]
    # Optional: use same pattern as /v1/optimize if you want callback-driven eval
    callback_url: Optional[str] = None
    dimension: Optional[int] = 1
    bounds: Optional[tuple[float, float]] = (0.0, 1.0)
    modules: Optional[List[str]] = None
    seed: Optional[int] = 123
    mode: str = "mo"
