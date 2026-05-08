from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import os
import logging
import secrets
import string
import uuid
import csv
import io
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal

import bcrypt
import jwt
from bson import ObjectId
from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Depends, Query
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fitest")

mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get("DB_NAME", "fitest_db")]

JWT_ALGORITHM = "HS256"
JWT_ACCESS_TTL_MIN = 60 * 24  # 1 day
JWT_REFRESH_TTL_DAYS = 7

app = FastAPI(title="Fitest API")
api_router = APIRouter(prefix="/api")

# ----------------------------------------------------------------------------
# Audit Question Bank
# ----------------------------------------------------------------------------
AUDIT_QUESTIONS = [
    {"id": 1, "category": "Mental Energy", "text": "I feel mentally fresh at the start of most days."},
    {"id": 2, "category": "Sleep", "text": "I sleep 7+ hours of quality sleep most nights."},
    {"id": 3, "category": "Energy", "text": "My energy stays steady throughout the day, without crashes."},
    {"id": 4, "category": "Movement", "text": "I move my body for at least 30 minutes daily."},
    {"id": 5, "category": "Stress", "text": "My stress feels manageable on a typical day."},
    {"id": 6, "category": "Recovery", "text": "I can switch off in the evenings without difficulty."},
    {"id": 7, "category": "Nutrition", "text": "I eat regular, nourishing meals."},
    {"id": 8, "category": "Nutrition", "text": "I do not rely on caffeine to get through the day."},
    {"id": 9, "category": "Nutrition", "text": "My blood sugar feels stable (no major hunger or shakiness)."},
    {"id": 10, "category": "Sleep", "text": "I fall asleep without anxiety or a racing mind."},
    {"id": 11, "category": "Hydration", "text": "I drink at least 2 litres of water each day."},
    {"id": 12, "category": "Nutrition", "text": "I rarely crave sugar or processed food."},
    {"id": 13, "category": "Recovery", "text": "I do not show signs of burnout."},
    {"id": 14, "category": "Recovery", "text": "I recover quickly after a bad night of sleep."},
    {"id": 15, "category": "Body Composition", "text": "My waist measurement has been stable over time."},
    {"id": 16, "category": "Lifestyle", "text": "I use my full annual leave each year."},
    {"id": 17, "category": "Focus", "text": "I can focus deeply on a single task for 60+ minutes."},
    {"id": 18, "category": "Digestion", "text": "I do not feel bloated or sluggish after meals."},
    {"id": 19, "category": "Work-Life Balance", "text": "I do not work late evenings or weekends regularly."},
    {"id": 20, "category": "Resilience", "text": "Overall, I feel energised and resilient."},
]

def calculate_score(answers: dict) -> int:
    if not answers: return 0
    total = sum(int(v) for v in answers.values())
    max_possible = len(AUDIT_QUESTIONS) * 5
    return max(0, min(100, round((total / max_possible) * 100)))

def calculate_tier(score: int) -> str:
    if score <= 25: return "Critical"
    if score <= 50: return "Exposed"
    if score <= 75: return "Performing"
    return "Elite"

# ----------------------------------------------------------------------------
# Auth Helpers
# ----------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

def get_jwt_secret() -> str:
    return os.environ.get("JWT_SECRET", "change-me-in-production")

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TTL_MIN),
        "type": "access",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_TTL_DAYS),
        "type": "refresh",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)

async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user: raise HTTPException(status_code=401, detail="User not found")
        user["id"] = str(user.pop("_id"))
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str
    company: Optional[str] = None
    license_type: Optional[Literal["gym", "business"]] = "gym"
    referred_by: Optional[str] = None

class UserOut(BaseModel):
    id: str
    email: EmailStr
    name: str
    company: Optional[str] = None
    role: str = "license_holder"
    license_type: Optional[str] = None
    license_active: bool = False

class AuditLinkCreate(BaseModel):
    label: str = "Default Audit"
    custom_slug: Optional[str] = None

class AuditSubmitIn(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    department: Optional[str] = None
    answers: dict
    opt_in: bool = False
    consent_medical: bool

class DeptAggregateOut(BaseModel):
    department: str
    total: int
    average_score: int
    tiers: dict

# ----------------------------------------------------------------------------
# Core Logic & Endpoints
# ----------------------------------------------------------------------------
@api_router.post("/auth/register", response_model=UserOut)
async def register(payload: RegisterIn, response: Response):
    email = payload.email.lower()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    doc = {
        "email": email,
        "password_hash": hash_password(payload.password),
        "name": payload.name,
        "company": payload.company,
        "role": "license_holder",
        "license_type": payload.license_type,
        "license_active": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "referred_by": payload.referred_by
    }
    result = await db.users.insert_one(doc)
    user_id = str(result.inserted_id)
    
    # Check for pending licenses from Stripe
    pending = await db.pending_licenses.find_one({"email": email})
    if pending:
        await db.users.update_one({"_id": result.inserted_id}, {"$set": {"license_active": True}})
        await db.pending_licenses.delete_one({"email": email})

    acc = create_access_token(user_id, email)
    ref = create_refresh_token(user_id)
    response.set_cookie("access_token", acc, httponly=True, secure=True, samesite="none")
    return {**doc, "id": user_id}

@api_router.get("/audit/{slug}")
async def get_audit(slug: str):
    link = await db.audit_links.find_one({"slug": slug})
    if not link: raise HTTPException(404, "Audit link not found")
    return {"slug": slug, "owner_name": link["owner_name"], "questions": AUDIT_QUESTIONS}

@api_router.post("/audit/{slug}/submit")
async def submit_audit(slug: str, payload: AuditSubmitIn):
    link = await db.audit_links.find_one({"slug": slug})
    if not link: raise HTTPException(404)
    score = calculate_score(payload.answers)
    tier = calculate_tier(score)
    doc = {
        "id": str(uuid.uuid4()),
        "owner_id": link["owner_id"],
        "score": score,
        "tier": tier,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        **payload.dict()
    }
    await db.leads
