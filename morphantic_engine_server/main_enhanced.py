"""
Morphantic Core API - Enhanced with User Authentication and Database Storage
Run with: uvicorn main_enhanced:app --reload
"""

from datetime import datetime, timedelta
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

import numpy as np
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from database import get_db, init_db, User, APIKey, UsageLog, OptimizationJob
from auth import (
    get_password_hash, verify_password, create_access_token, create_refresh_token,
    verify_token, get_current_user, generate_api_key, hash_api_key,
    validate_api_key, get_current_user_or_api_key, validate_password_strength
)
from schemas import (
    UserSignup, UserLogin, UserResponse, TokenResponse, RefreshTokenRequest,
    CreateAPIKeyRequest, APIKeyResponse, APIKeyInfo, UsageStats,
    PasswordResetRequest, PasswordResetConfirm, ChangePasswordRequest,
    UpdateUserProfile
)
from pydantic import BaseModel, Field, validator

from complete_teai_methods_slim_v2 import AdvancedArchipelagoEvolution, ObjectiveSpec
from morphantic_modules import attach_modules

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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

async def log_usage(
    request: Request,
    user: Optional[User],
    api_key: Optional[APIKey],
    status_code: int,
    response_time_ms: int,
    db: Session
):
    """Log API usage for analytics"""
    if user:
        usage_log = UsageLog(
            user_id=user.id,
            api_key_id=api_key.id if api_key else None,
            endpoint=str(request.url.path),
            method=request.method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )
        db.add(usage_log)
        db.commit()

@app.post("/v1/auth/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Register a new user account"""
    
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    is_valid, message = validate_password_strength(user_data.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    user = User(
        email=user_data.email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        hashed_password=get_password_hash(user_data.password)
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user

@app.post("/v1/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and receive JWT tokens"""
    
    user = db.query(User).filter(User.email == credentials.email).first()
    
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
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh access token using refresh token"""
    
    payload = verify_token(request.refresh_token, "refresh")
    user_id = payload.get("sub")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    access_token = create_access_token(data={"sub": str(user.id)})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=request.refresh_token,
        expires_in=1800
    )

@app.get("/v1/auth/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@app.put("/v1/auth/me", response_model=UserResponse)
async def update_user_profile(
    update_data: UpdateUserProfile,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    
    if update_data.first_name:
        current_user.first_name = update_data.first_name
    if update_data.last_name:
        current_user.last_name = update_data.last_name
    
    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    
    return current_user

@app.post("/v1/auth/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
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
    
    current_user.hashed_password = get_password_hash(request.new_password)
    current_user.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Password changed successfully"}

@app.post("/v1/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key for the authenticated user"""
    
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    
    expires_at = None
    if request.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_days)
    
    api_key_obj = APIKey(
        key_hash=key_hash,
        user_id=current_user.id,
        name=request.name,
        permissions=request.permissions or ["optimize", "read_results"],
        expires_at=expires_at,
        rate_limit=request.rate_limit or 100
    )
    
    db.add(api_key_obj)
    db.commit()
    db.refresh(api_key_obj)
    
    return APIKeyResponse(
        api_key=api_key,
        id=api_key_obj.id,
        name=api_key_obj.name,
        expires_at=api_key_obj.expires_at,
        permissions=api_key_obj.permissions,
        created_at=api_key_obj.created_at
    )

@app.get("/v1/api-keys", response_model=List[APIKeyInfo])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all API keys for the authenticated user"""
    
    api_keys = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.is_active == True
    ).all()
    
    return api_keys

@app.delete("/v1/api-keys/{key_id}")
async def revoke_api_key(
    key_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke a specific API key"""
    
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    api_key.is_active = False
    db.commit()
    
    return {"message": "API key revoked successfully"}

@app.get("/v1/usage/stats", response_model=UsageStats)
async def get_usage_stats(
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db)
):
    """Get usage statistics for the authenticated user"""
    
    user = auth["user"]
    now = datetime.utcnow()
    
    total = db.query(func.count(UsageLog.id)).filter(UsageLog.user_id == user.id).scalar()
    successful = db.query(func.count(UsageLog.id)).filter(
        UsageLog.user_id == user.id,
        UsageLog.status_code.between(200, 299)
    ).scalar()
    
    avg_response_time = db.query(func.avg(UsageLog.response_time_ms)).filter(
        UsageLog.user_id == user.id
    ).scalar() or 0
    
    last_24h = db.query(func.count(UsageLog.id)).filter(
        UsageLog.user_id == user.id,
        UsageLog.created_at >= now - timedelta(days=1)
    ).scalar()
    
    last_7d = db.query(func.count(UsageLog.id)).filter(
        UsageLog.user_id == user.id,
        UsageLog.created_at >= now - timedelta(days=7)
    ).scalar()
    
    last_30d = db.query(func.count(UsageLog.id)).filter(
        UsageLog.user_id == user.id,
        UsageLog.created_at >= now - timedelta(days=30)
    ).scalar()
    
    return UsageStats(
        total_requests=total,
        successful_requests=successful,
        failed_requests=total - successful,
        average_response_time_ms=avg_response_time,
        requests_last_24h=last_24h,
        requests_last_7d=last_7d,
        requests_last_30d=last_30d
    )

class OptimizerConfig(BaseModel):
    pop_size: int = Field(60, gt=0, le=500)
    max_generations: int = Field(45, gt=0, le=1000)
    n_islands: int = Field(4, gt=0, le=16)
    seed: Optional[int] = None

class OptimizationRequest(BaseModel):
    objectives: Optional[List[ObjectiveSpec]] = Field(None, description="Custom objectives")
    callback_url: str
    dimension: int = Field(..., gt=0, le=2048)
    bounds: Tuple[float, float]
    config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    modules: Optional[List[str]] = Field(None, description="List of module specs")
    domain: Optional[str] = Field(None, description="Specialty domain")

    @validator('objectives', always=True)
    def check_objectives_or_domain(cls, v, values):
        if not values.get('domain') and not v:
            raise ValueError("Either 'domain' or 'objectives' must be provided")
        return v

def run_optimization_task(
    job_id: str, 
    request: OptimizationRequest, 
    user_id: str, 
    db_session: Session
):
    """Background task for running optimization"""
    
    job = db_session.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()
    if not job:
        return
    
    job.status = "running"
    job.started_at = datetime.utcnow()
    db_session.commit()
    
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
                response = requests.post(
                    request.callback_url,
                    json={'solution': x.tolist()},
                    timeout=60
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                return {obj.name: obj.baseline for obj in final_objectives}
        
        aea = AdvancedArchipelagoEvolution(
            metrics_fn=metrics_fn_via_callback,
            objectives=final_objectives,
            dimension=request.dimension,
            bounds=request.bounds,
            pop_size=request.config.pop_size,
            max_generations=request.config.max_generations,
            n_islands=request.config.n_islands,
            seed=request.config.seed
        )
        
        final_fitness_func = attach_modules(
            inner_fitness=aea.current_fitness_func,
            dim=request.dimension,
            bounds=request.bounds,
            budget_max=(request.config.pop_size * request.config.max_generations),
            seed=request.config.seed,
            modules=final_modules,
            problem_name=request.domain or "custom"
        )
        
        result, best_cell = aea.optimize(fitness_func=final_fitness_func)
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.result = {
            "best_solution": best_cell.get_solution().tolist(),
            "best_fitness": result.final_fitness,
            "final_metrics": aea.last_metrics
        }
        
    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        job.error_message = str(e)
    
    db_session.commit()

@app.post("/v1/optimize", status_code=status.HTTP_202_ACCEPTED)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db)
):
    """Start an optimization job"""
    
    user = auth["user"]
    
    if auth["type"] == "api_key" and "optimize" not in auth["api_key"].permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key lacks optimize permission"
        )
    
    job = OptimizationJob(
        user_id=user.id,
        status="queued",
        domain=request.domain,
        config=request.config.dict()
    )
    
    db.add(job)
    db.commit()
    db.refresh(job)
    
    background_tasks.add_task(
        run_optimization_task,
        str(job.id),
        request,
        str(user.id),
        db
    )
    
    return {"job_id": str(job.id), "message": "Optimization job started"}

@app.get("/v1/results/{job_id}")
async def get_job_results(
    job_id: uuid.UUID,
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db)
):
    """Get optimization job results"""
    
    user = auth["user"]
    
    job = db.query(OptimizationJob).filter(
        OptimizationJob.id == job_id,
        OptimizationJob.user_id == user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return {
        "job_id": str(job.id),
        "status": job.status,
        "domain": job.domain,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": job.result,
        "error_message": job.error_message
    }

@app.get("/v1/jobs")
async def list_jobs(
    auth: Dict[str, Any] = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db),
    limit: int = 10,
    offset: int = 0
):
    """List optimization jobs for the authenticated user"""
    
    user = auth["user"]
    
    jobs = db.query(OptimizationJob).filter(
        OptimizationJob.user_id == user.id
    ).order_by(OptimizationJob.started_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "jobs": [
            {
                "job_id": str(job.id),
                "status": job.status,
                "domain": job.domain,
                "started_at": job.started_at,
                "completed_at": job.completed_at
            }
            for job in jobs
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)