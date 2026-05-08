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

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

JWT_ALGORITHM = "HS256"
JWT_ACCESS_TTL_MIN = 60 * 24  # 1 day for convenience
JWT_REFRESH_TTL_DAYS = 7

app = FastAPI(title="Fitest API")
api_router = APIRouter(prefix="/api")


# ----------------------------------------------------------------------------
# Audit Question Bank (defaults — user will provide exact list later)
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
    """Each answer is 1-5. Convert to 0-100."""
    if not answers:
        return 0
    total = sum(int(v) for v in answers.values())
    max_possible = len(AUDIT_QUESTIONS) * 5
    pct = round((total / max_possible) * 100)
    return max(0, min(100, pct))


def calculate_tier(score: int) -> str:
    if score <= 25:
        return "Critical"
    if score <= 50:
        return "Exposed"
    if score <= 75:
        return "Performing"
    return "Elite"


# ----------------------------------------------------------------------------
# Auth helpers
# ----------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def get_jwt_secret() -> str:
    return os.environ["JWT_SECRET"]


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


def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    response.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=JWT_ACCESS_TTL_MIN * 60,
        path="/",
    )
    response.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=JWT_REFRESH_TTL_DAYS * 86400,
        path="/",
    )


def clear_auth_cookies(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")


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
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        user["id"] = str(user.pop("_id"))
        user.pop("password_hash", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ----------------------------------------------------------------------------
# Pydantic Models
# ----------------------------------------------------------------------------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str
    company: Optional[str] = None
    license_type: Optional[Literal["gym", "business"]] = "gym"
    referred_by: Optional[str] = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str


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
    custom_slug: Optional[str] = None  # e.g. "city-gym-london"


class AuditLinkOut(BaseModel):
    id: str
    slug: str
    label: str
    created_at: str
    submission_count: int = 0


class AuditPublic(BaseModel):
    slug: str
    owner_name: str
    owner_company: Optional[str] = None
    questions: List[dict]


class AuditSubmitIn(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    department: Optional[str] = None  # for Business license — anonymised aggregation
    answers: dict  # {question_id_str: 1..5}
    opt_in: bool = False
    consent_medical: bool


class AuditSubmitOut(BaseModel):
    score: int
    tier: str
    submission_id: str


class LeadOut(BaseModel):
    id: str
    name: str
    email: Optional[str] = None  # hidden unless opt_in
    email_hidden: bool
    score: int
    tier: str
    opt_in: bool
    department: Optional[str] = None
    submitted_at: str


class DeptAggregateOut(BaseModel):
    department: str
    total: int
    average_score: int
    tiers: dict  # { Critical: int, Exposed: int, Performing: int, Elite: int }


# ----------------------------------------------------------------------------
# Auth Endpoints
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
    }
    result = await db.users.insert_one(doc)
    user_id = str(result.inserted_id)

    # Track referral if a referrer user_id was passed in
    if payload.referred_by:
        try:
            ref_user = await db.users.find_one({"_id": ObjectId(payload.referred_by)}, {"_id": 1})
            if ref_user:
                await db.users.update_one(
                    {"_id": result.inserted_id},
                    {"$set": {"referred_by": str(ref_user["_id"])}},
                )
        except Exception:
            pass  # invalid ref id — silently ignore

    # Merge any pending license that was created via Stripe webhook before registration
    pending = await db.pending_licenses.find_one({"email": email})
    license_active = False
    if pending:
        await db.users.update_one(
            {"_id": result.inserted_id},
            {
                "$set": {
                    "license_active": True,
                    "license_type": pending.get("license_type", payload.license_type),
                    "license_activated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        license_active = True
        # Provision the audit link reserved for them at checkout time
        slug = pending.get("audit_slug")
        if slug and not await db.audit_links.find_one({"slug": slug}):
            await db.audit_links.insert_one(
                {
                    "id": str(uuid.uuid4()),
                    "slug": slug,
                    "label": "Welcome — your first audit link",
                    "owner_id": user_id,
                    "owner_name": payload.name,
                    "owner_company": payload.company,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "submission_count": 0,
                }
            )
        await db.pending_licenses.delete_one({"email": email})

    set_auth_cookies(response, create_access_token(user_id, email), create_refresh_token(user_id))
    return UserOut(
        id=user_id,
        email=email,
        name=payload.name,
        company=payload.company,
        role="license_holder",
        license_type=payload.license_type,
        license_active=license_active,
    )


@api_router.post("/auth/login", response_model=UserOut)
async def login(payload: LoginIn, response: Response):
    email = payload.email.lower()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user_id = str(user["_id"])
    set_auth_cookies(response, create_access_token(user_id, email), create_refresh_token(user_id))
    return UserOut(
        id=user_id,
        email=user["email"],
        name=user.get("name", ""),
        company=user.get("company"),
        role=user.get("role", "license_holder"),
        license_type=user.get("license_type"),
        license_active=user.get("license_active", False),
    )


@api_router.post("/auth/logout")
async def logout(response: Response, _user: dict = Depends(get_current_user)):
    clear_auth_cookies(response)
    return {"ok": True}


@api_router.get("/auth/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)):
    return UserOut(
        id=user["id"],
        email=user["email"],
        name=user.get("name", ""),
        company=user.get("company"),
        role=user.get("role", "license_holder"),
        license_type=user.get("license_type"),
        license_active=user.get("license_active", False),
    )


# ----------------------------------------------------------------------------
# Audit Public Endpoints
# ----------------------------------------------------------------------------
@api_router.get("/audit/questions", response_model=List[dict])
async def get_audit_questions():
    return AUDIT_QUESTIONS


def _gen_slug(length: int = 8) -> str:
    alpha = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alpha) for _ in range(length))


def _slugify(value: str) -> str:
    """Convert a free-form string into a URL-safe slug."""
    out = []
    for ch in value.lower().strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "/"):
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48]


@api_router.post("/audit-links", response_model=AuditLinkOut)
async def create_audit_link(payload: AuditLinkCreate, user: dict = Depends(get_current_user)):
    # If user supplied a custom slug, normalise and check uniqueness
    if payload.custom_slug:
        slug = _slugify(payload.custom_slug)
        if not slug or len(slug) < 3:
            raise HTTPException(400, "Custom slug must be at least 3 valid characters (a-z, 0-9, -)")
        if await db.audit_links.find_one({"slug": slug}):
            raise HTTPException(409, f"Slug '{slug}' is already taken")
    else:
        for _ in range(10):
            slug = _gen_slug()
            if not await db.audit_links.find_one({"slug": slug}):
                break
        else:
            raise HTTPException(500, "Could not generate slug")
    doc = {
        "id": str(uuid.uuid4()),
        "slug": slug,
        "label": payload.label,
        "owner_id": user["id"],
        "owner_name": user.get("name", ""),
        "owner_company": user.get("company"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_count": 0,
    }
    await db.audit_links.insert_one(doc)
    return AuditLinkOut(
        id=doc["id"],
        slug=slug,
        label=doc["label"],
        created_at=doc["created_at"],
        submission_count=0,
    )


@api_router.get("/audit-links/me", response_model=List[AuditLinkOut])
async def list_my_audit_links(user: dict = Depends(get_current_user)):
    cursor = db.audit_links.find({"owner_id": user["id"]}, {"_id": 0}).sort("created_at", -1)
    items = await cursor.to_list(500)
    return [
        AuditLinkOut(
            id=i["id"],
            slug=i["slug"],
            label=i["label"],
            created_at=i["created_at"],
            submission_count=i.get("submission_count", 0),
        )
        for i in items
    ]


@api_router.get("/audit/{slug}", response_model=AuditPublic)
async def get_audit(slug: str):
    link = await db.audit_links.find_one({"slug": slug}, {"_id": 0})
    if not link:
        raise HTTPException(404, "Audit not found")
    return AuditPublic(
        slug=slug,
        owner_name=link.get("owner_name") or "Fitest",
        owner_company=link.get("owner_company"),
        questions=AUDIT_QUESTIONS,
    )


@api_router.post("/audit/{slug}/submit", response_model=AuditSubmitOut)
async def submit_audit(slug: str, payload: AuditSubmitIn):
    if not payload.consent_medical:
        raise HTTPException(400, "Medical disclaimer must be accepted")
    link = await db.audit_links.find_one({"slug": slug}, {"_id": 0})
    if not link:
        raise HTTPException(404, "Audit not found")
    score = calculate_score(payload.answers)
    tier = calculate_tier(score)
    submission_id = str(uuid.uuid4())
    doc = {
        "id": submission_id,
        "slug": slug,
        "owner_id": link["owner_id"],
        "name": payload.name,
        "email": payload.email.lower(),
        "phone": payload.phone,
        "department": (payload.department or "").strip() or None,
        "answers": payload.answers,
        "score": score,
        "tier": tier,
        "opt_in": payload.opt_in,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.leads.insert_one(doc)
    await db.audit_links.update_one({"slug": slug}, {"$inc": {"submission_count": 1}})
    return AuditSubmitOut(score=score, tier=tier, submission_id=submission_id)


# ----------------------------------------------------------------------------
# Leads & Reports
# ----------------------------------------------------------------------------
def _lead_to_out(d: dict, reveal: bool = False) -> LeadOut:
    return LeadOut(
        id=d["id"],
        name=d["name"],
        email=d["email"] if (d.get("opt_in") or reveal) else None,
        email_hidden=not (d.get("opt_in") or reveal),
        score=d["score"],
        tier=d["tier"],
        opt_in=d.get("opt_in", False),
        department=d.get("department"),
        submitted_at=d["submitted_at"],
    )


@api_router.get("/leads", response_model=List[LeadOut])
async def list_leads(
    user: dict = Depends(get_current_user),
    days: Optional[int] = Query(default=None, description="Filter to last N days"),
    tier: Optional[str] = None,
):
    q = {"owner_id": user["id"]}
    if tier:
        q["tier"] = tier
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        q["submitted_at"] = {"$gte": cutoff}
    cursor = db.leads.find(q, {"_id": 0}).sort("submitted_at", -1)
    items = await cursor.to_list(2000)
    return [_lead_to_out(i) for i in items]


@api_router.get("/leads/{lead_id}/email")
async def reveal_email(lead_id: str, user: dict = Depends(get_current_user)):
    lead = await db.leads.find_one({"id": lead_id, "owner_id": user["id"]}, {"_id": 0})
    if not lead:
        raise HTTPException(404, "Lead not found")
    if not lead.get("opt_in"):
        raise HTTPException(403, "Lead did not opt in for outreach")
    return {"email": lead["email"]}


@api_router.get("/reports/csv")
async def report_csv(days: int = Query(7), user: dict = Depends(get_current_user)):
    if days not in (7, 14, 30):
        raise HTTPException(400, "days must be 7, 14, or 30")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cursor = db.leads.find(
        {"owner_id": user["id"], "submitted_at": {"$gte": cutoff}}, {"_id": 0}
    ).sort("submitted_at", -1)
    items = await cursor.to_list(5000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Submitted At", "Name", "Email (if opt-in)", "Score", "Tier", "Opt-in", "Phone"])
    for i in items:
        writer.writerow(
            [
                i["submitted_at"],
                i["name"],
                i["email"] if i.get("opt_in") else "",
                i["score"],
                i["tier"],
                "Yes" if i.get("opt_in") else "No",
                i.get("phone") or "",
            ]
        )
    output.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="fitest-report-{days}d.csv"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)


@api_router.get("/reports/summary")
async def report_summary(days: int = Query(30), user: dict = Depends(get_current_user)):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cursor = db.leads.find(
        {"owner_id": user["id"], "submitted_at": {"$gte": cutoff}}, {"_id": 0}
    )
    items = await cursor.to_list(5000)
    counts = {"Critical": 0, "Exposed": 0, "Performing": 0, "Elite": 0}
    total_score = 0
    for i in items:
        counts[i["tier"]] = counts.get(i["tier"], 0) + 1
        total_score += i["score"]
    avg = round(total_score / len(items)) if items else 0
    return {
        "total": len(items),
        "average_score": avg,
        "tiers": counts,
        "opt_in_count": sum(1 for i in items if i.get("opt_in")),
    }


@api_router.get("/reports/departments", response_model=List[DeptAggregateOut])
async def report_departments(
    days: Optional[int] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    """
    Anonymised aggregation by department — used by Business License holders.
    Never returns names or emails — only counts and tier mix.
    """
    q = {"owner_id": user["id"]}
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        q["submitted_at"] = {"$gte": cutoff}
    cursor = db.leads.find(q, {"_id": 0})
    items = await cursor.to_list(10000)

    buckets: dict = {}
    for i in items:
        dept = (i.get("department") or "Unassigned").strip() or "Unassigned"
        b = buckets.setdefault(
            dept,
            {"total": 0, "score_sum": 0, "tiers": {"Critical": 0, "Exposed": 0, "Performing": 0, "Elite": 0}},
        )
        b["total"] += 1
        b["score_sum"] += int(i.get("score", 0))
        b["tiers"][i["tier"]] = b["tiers"].get(i["tier"], 0) + 1
    return [
        DeptAggregateOut(
            department=dept,
            total=b["total"],
            average_score=round(b["score_sum"] / b["total"]) if b["total"] else 0,
            tiers=b["tiers"],
        )
        for dept, b in sorted(buckets.items(), key=lambda kv: -kv[1]["total"])
    ]


# ----------------------------------------------------------------------------
# Affiliate Portal (20% recurring on annual licences)
# ----------------------------------------------------------------------------
LICENSE_ANNUAL = {"gym": 149.0, "business": 249.0}  # GBP / year
COMMISSION_RATE = 0.20


@api_router.post("/affiliate/track-click")
async def track_click(payload: dict):
    """Anonymous click tracking — called from frontend when ?ref=USER_ID is present."""
    ref = (payload.get("ref") or "").strip()
    if not ref:
        return {"ok": False}
    try:
        ref_user = await db.users.find_one({"_id": ObjectId(ref)}, {"_id": 1})
    except Exception:
        return {"ok": False}
    if not ref_user:
        return {"ok": False}
    await db.affiliate_clicks.insert_one(
        {
            "id": str(uuid.uuid4()),
            "ref_user_id": str(ref_user["_id"]),
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {"ok": True}


async def _compute_affiliate_stats(user_id: str) -> dict:
    clicks = await db.affiliate_clicks.count_documents({"ref_user_id": user_id})
    signups_cursor = db.users.find(
        {"referred_by": user_id},
        {"_id": 1, "license_type": 1, "license_active": 1, "created_at": 1},
    )
    signups = await signups_cursor.to_list(5000)
    active_signups = [s for s in signups if s.get("license_active")]

    # Annual commission per active referral
    annual_commission = sum(
        LICENSE_ANNUAL.get(s.get("license_type", "gym"), 149.0) * COMMISSION_RATE
        for s in active_signups
    )

    # Lifetime earnings — sum (annual_price * 20%) per year the licence has been active
    lifetime = 0.0
    now = datetime.now(timezone.utc)
    for s in active_signups:
        try:
            created = datetime.fromisoformat(s.get("created_at", now.isoformat()))
        except Exception:
            created = now
        years_active = max(1, (now - created).days // 365 + 1)
        lifetime += LICENSE_ANNUAL.get(s.get("license_type", "gym"), 149.0) * COMMISSION_RATE * years_active

    paid_cursor = db.affiliate_withdrawals.find(
        {"user_id": user_id, "status": {"$in": ["pending", "paid"]}}, {"_id": 0, "amount": 1}
    )
    paid_items = await paid_cursor.to_list(1000)
    paid_total = sum(p.get("amount", 0) for p in paid_items)
    balance = max(0.0, lifetime - paid_total)

    return {
        "referral_url_path": f"/?ref={user_id}",
        "user_id": user_id,
        "clicks": clicks,
        "signups": len(signups),
        "active_signups": len(active_signups),
        "annual_commission": round(annual_commission, 2),
        # kept for backwards-compat with iter_2 frontend until rolled — same value
        "monthly_commission": round(annual_commission, 2),
        "lifetime_earnings": round(lifetime, 2),
        "withdrawn_to_date": round(paid_total, 2),
        "available_balance": round(balance, 2),
        "commission_rate": COMMISSION_RATE,
        "billing_cycle": "annual",
        "payout_schedule": "Payouts are processed manually on the 1st of every month via Stripe.",
    }


@api_router.get("/affiliate/stats")
async def affiliate_stats(user: dict = Depends(get_current_user)):
    return await _compute_affiliate_stats(user["id"])


@api_router.post("/affiliate/withdraw")
async def affiliate_withdraw(user: dict = Depends(get_current_user)):
    """
    MOCKED Stripe Express payout — creates a pending withdrawal record.
    Real implementation requires Stripe Connect onboarding for the affiliate.
    """
    stats = await _compute_affiliate_stats(user["id"])
    balance = stats["available_balance"]
    if balance <= 0:
        raise HTTPException(400, "No available balance to withdraw.")

    record = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "email": user["email"],
        "amount": balance,
        "currency": "GBP",
        "status": "pending",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "scheduled_for": "Manual payout — 1st of next month via Stripe",
        "processing_mode": "manual_ledger",
    }
    await db.affiliate_withdrawals.insert_one(record)
    record.pop("_id", None)
    return {"ok": True, "withdrawal": record}


# ----------------------------------------------------------------------------
# License activation + audit seeding (Stripe webhook helper)
# ----------------------------------------------------------------------------
async def _activate_license_and_seed_audit(email: str, license_type: str, name: Optional[str] = None):
    """
    Triggered after a successful Stripe checkout.session.completed.
    - If the user already exists -> activate license + ensure they have an audit slug.
    - If the user does not yet exist -> store a pending license that gets merged on register.
    Always tries to mint a unique audit slug derived from email so they have something
    to share immediately.
    """
    email = email.lower()
    existing = await db.users.find_one({"email": email})

    desired_slug_base = _slugify(email.split("@")[0]) or _gen_slug()
    slug = desired_slug_base
    suffix = 0
    while await db.audit_links.find_one({"slug": slug}):
        suffix += 1
        slug = f"{desired_slug_base}-{suffix}"

    if existing:
        await db.users.update_one(
            {"email": email},
            {
                "$set": {
                    "license_active": True,
                    "license_type": license_type,
                    "license_activated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        # Only create a default audit link if they don't have one yet
        has_link = await db.audit_links.find_one({"owner_id": str(existing["_id"])})
        if not has_link:
            await db.audit_links.insert_one(
                {
                    "id": str(uuid.uuid4()),
                    "slug": slug,
                    "label": "Welcome — your first audit link",
                    "owner_id": str(existing["_id"]),
                    "owner_name": existing.get("name", ""),
                    "owner_company": existing.get("company"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "submission_count": 0,
                }
            )
    else:
        await db.pending_licenses.update_one(
            {"email": email},
            {
                "$set": {
                    "email": email,
                    "license_type": license_type,
                    "name": name,
                    "audit_slug": slug,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )


# ----------------------------------------------------------------------------
# Stripe webhook (license activation)
# ----------------------------------------------------------------------------
@api_router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Listens for checkout.session.completed events from Stripe Payment Links.
    Activates license for the customer email if matched in our users collection.
    """
    try:
        from emergentintegrations.payments.stripe.checkout import StripeCheckout

        api_key = os.environ.get("STRIPE_API_KEY", "sk_test_emergent")
        host_url = str(request.base_url)
        webhook_url = f"{host_url}api/webhook/stripe"
        sc = StripeCheckout(api_key=api_key, webhook_url=webhook_url)
        body = await request.body()
        sig = request.headers.get("Stripe-Signature", "")
        event = await sc.handle_webhook(body, sig)

        if event.event_type == "checkout.session.completed":
            metadata = event.metadata or {}
            email = (metadata.get("email") or "").lower()
            license_type = metadata.get("license_type") or "gym"
            await db.payment_transactions.insert_one(
                {
                    "id": str(uuid.uuid4()),
                    "session_id": event.session_id,
                    "event_id": event.event_id,
                    "email": email,
                    "license_type": license_type,
                    "payment_status": event.payment_status,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if email:
                await _activate_license_and_seed_audit(email, license_type, metadata.get("name"))
        return {"received": True}
    except Exception as e:
        logger.exception("Stripe webhook error: %s", e)
        # Return 200 anyway to prevent Stripe retries during dev
        return {"received": False, "error": str(e)}


# ----------------------------------------------------------------------------
# Admin (Withdrawal Requests ledger)
# ----------------------------------------------------------------------------
def _require_admin(user: dict):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


@api_router.get("/admin/withdrawals")
async def admin_list_withdrawals(
    status: Optional[str] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    _require_admin(user)
    q: dict = {}
    if status and status != "all":
        q["status"] = status
    cursor = db.affiliate_withdrawals.find(q, {"_id": 0}).sort("requested_at", -1)
    items = await cursor.to_list(2000)
    return items


@api_router.post("/admin/withdrawals/{withdrawal_id}/mark-paid")
async def admin_mark_paid(withdrawal_id: str, user: dict = Depends(get_current_user)):
    _require_admin(user)
    record = await db.affiliate_withdrawals.find_one({"id": withdrawal_id}, {"_id": 0})
    if not record:
        raise HTTPException(404, "Withdrawal request not found")
    if record.get("status") == "paid":
        raise HTTPException(400, "Already marked as paid")
    await db.affiliate_withdrawals.update_one(
        {"id": withdrawal_id},
        {
            "$set": {
                "status": "paid",
                "paid_at": datetime.now(timezone.utc).isoformat(),
                "paid_by": user["email"],
            }
        },
    )
    updated = await db.affiliate_withdrawals.find_one({"id": withdrawal_id}, {"_id": 0})
    return {"ok": True, "withdrawal": updated}


# ----------------------------------------------------------------------------
# Renewal Saver — usage stats + days-until-renewal for the current licence holder
# ----------------------------------------------------------------------------
@api_router.get("/renewal/usage")
async def renewal_usage(user: dict = Depends(get_current_user)):
    user_full = await db.users.find_one({"_id": ObjectId(user["id"])})
    if not user_full:
        raise HTTPException(404, "User not found")

    activated_iso = user_full.get("license_activated_at") or user_full.get("created_at")
    try:
        activated = datetime.fromisoformat(activated_iso) if activated_iso else datetime.now(timezone.utc)
    except Exception:
        activated = datetime.now(timezone.utc)
    if activated.tzinfo is None:
        activated = activated.replace(tzinfo=timezone.utc)

    renewal_at = activated + timedelta(days=365)
    days_until = max(0, (renewal_at - datetime.now(timezone.utc)).days)

    # Audits completed in the trailing licence year
    cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    leads_cursor = db.leads.find(
        {"owner_id": user["id"], "submitted_at": {"$gte": cutoff}}, {"_id": 0}
    )
    leads = await leads_cursor.to_list(20000)

    tier_counts = {"Critical": 0, "Exposed": 0, "Performing": 0, "Elite": 0}
    opt_in = 0
    score_total = 0
    for lead in leads:
        tier_counts[lead.get("tier", "Performing")] = tier_counts.get(lead.get("tier", "Performing"), 0) + 1
        if lead.get("opt_in"):
            opt_in += 1
        score_total += int(lead.get("score", 0))

    audits_completed = len(leads)
    avg_score = round(score_total / audits_completed) if audits_completed else 0
    healthy_count = tier_counts["Performing"] + tier_counts["Elite"]
    at_risk_count = tier_counts["Critical"] + tier_counts["Exposed"]

    license_type = user_full.get("license_type", "gym")
    renew_url = (
        "https://buy.stripe.com/8x29ASdLL99z3ycbMX6AM05"
        if license_type == "gym"
        else "https://buy.stripe.com/fZudR89vvdpP9WA9EP6AM06"
    )

    return {
        "license_type": license_type,
        "license_active": user_full.get("license_active", False),
        "activated_at": activated.isoformat(),
        "renewal_at": renewal_at.isoformat(),
        "days_until_renewal": days_until,
        "show_renewal_prompt": days_until <= 30 and user_full.get("license_active", False),
        "audits_completed": audits_completed,
        "average_score": avg_score,
        "tiers": tier_counts,
        "opt_in_count": opt_in,
        "healthy_count": healthy_count,
        "at_risk_count": at_risk_count,
        "renew_url": renew_url,
    }


# ----------------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------------
@api_router.get("/")
async def root():
    return {"service": "fitest", "ok": True}


# ----------------------------------------------------------------------------
# Include router & middleware
# ----------------------------------------------------------------------------
app.include_router(api_router)

cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    cors_kwargs = {"allow_origin_regex": ".*", "allow_credentials": True}
else:
    cors_kwargs = {"allow_origins": [o.strip() for o in cors_origins_env.split(",")], "allow_credentials": True}

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    **cors_kwargs,
)


@app.on_event("startup")
async def on_startup():
    await db.users.create_index("email", unique=True)
    await db.audit_links.create_index("slug", unique=True)
    await db.leads.create_index("owner_id")
    await db.leads.create_index("submitted_at")

    admin_email = os.environ.get("ADMIN_EMAIL", "admin@fitest.co.uk").lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "FitestAdmin2026!")
    existing = await db.users.find_one({"email": admin_email})
    if existing is None:
        await db.users.insert_one(
            {
                "email": admin_email,
                "password_hash": hash_password(admin_password),
                "name": "Fitest Admin",
                "company": "Fitest",
                "role": "admin",
                "license_type": "business",
                "license_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("Seeded admin user: %s", admin_email)
    elif not verify_password(admin_password, existing["password_hash"]):
        await db.users.update_one(
            {"email": admin_email},
            {"$set": {"password_hash": hash_password(admin_password)}},
        )
        logger.info("Updated admin password for: %s", admin_email)


@app.on_event("shutdown")
async def on_shutdown():
    client.close()
