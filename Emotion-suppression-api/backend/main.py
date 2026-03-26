"""
Emotion Suppression Detection API
----------------------------------
Endpoint summary
  GET  /                        – health check
  POST /auth/signup             – create a new account
  POST /auth/login              – login, returns JWT token
  GET  /me                      – get current logged-in user (auth required)
  POST /register-user           – save / update subject details in MongoDB
  POST /analyze-video           – upload video, run pipeline, optionally save results
  GET  /my-results              – get all analysis results for current user (auth required)
  GET  /users                   – get all users + their results (auth required)
  GET  /users/{email}/results   – get analysis results for a specific user (auth required)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from suppression.logic import run_video_pipeline
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Optional
from bson import ObjectId
from bson.errors import InvalidId
from passlib.context import CryptContext
from jose import JWTError, jwt
import tempfile
import os

app = FastAPI(title="Emotion Suppression Detection API")

# ── JWT config ─────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET", "suppresense-superSecretKey-2024-do-not-expose")
ALGORITHM  = "HS256"
TOKEN_TTL_MINUTES = 60 * 24   # 24 hours

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://UserData:TeraPass@cluster0.n7jxgnt.mongodb.net/?appName=Cluster0",
)
mongo_client        = MongoClient(MONGO_URI)
db                  = mongo_client["emotion_suppression"]
users_collection    = db["users"]       # subject details (name/email/age/gender)
accounts_collection = db["accounts"]    # login credentials
results_collection  = db["results"]     # per-user analysis results

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
        # "https://suppression.netlify.app/",
        # "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth helpers ───────────────────────────────────────────────────────────────
def _verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False

def _hash_password(password: str) -> str:
    return pwd_context.hash(password)

def _normalize_role(role: Optional[str]) -> str:
    return "admin" if (role or "").lower() == "admin" else "user"

def _create_token(email: str, role: str = "user") -> str:
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_TTL_MINUTES)
    payload = {"sub": email, "role": _normalize_role(role), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _serialize_result(doc: dict, include_details: bool = False) -> dict:
    item = {
        "result_id": str(doc.get("_id")) if doc.get("_id") else None,
        "email": doc.get("email"),
        "file_name": doc.get("file_name"),
        "suppression_score": doc.get("suppression_score"),
        "normalized_score": doc.get("normalized_score"),
        "level": doc.get("level"),
        "dominant_emotion": doc.get("dominant_emotion"),
        "suppressed_emotion": doc.get("suppressed_emotion"),
        "files_processed": doc.get("files_processed", 1),
        "created_at": doc.get("created_at"),
    }
    if include_details:
        item["timeline"] = doc.get("timeline", [])
        item["latency_events"] = doc.get("latency_events", [])
    return item

def _resolve_user_email(user_id: str) -> Optional[str]:
    if "@" in user_id:
        return user_id

    try:
        obj_id = ObjectId(user_id)
    except (InvalidId, TypeError):
        return None

    acct = accounts_collection.find_one({"_id": obj_id}, {"email": 1})
    if acct and acct.get("email"):
        return acct["email"]

    profile = users_collection.find_one({"_id": obj_id}, {"email": 1})
    if profile and profile.get("email"):
        return profile["email"]

    return None

def _is_admin(user: dict) -> bool:
    return _normalize_role(user.get("role")) == "admin"

def _require_owner_or_admin(current_user: dict, target_email: str):
    if _is_admin(current_user):
        return
    if (current_user.get("email") or "").lower() != (target_email or "").lower():
        raise HTTPException(status_code=403, detail="Forbidden: cannot access other users' results")

def _list_results_for_email(email: str, include_details: bool = False) -> list[dict]:
    projection = {"timeline": 0, "latency_events": 0} if not include_details else None
    cursor = results_collection.find({"email": email}, projection).sort("created_at", -1)
    return [_serialize_result(doc, include_details=include_details) for doc in cursor]

def _find_result_by_id(result_id: str) -> Optional[dict]:
    try:
        obj_id = ObjectId(result_id)
    except (InvalidId, TypeError):
        return None
    return results_collection.find_one({"_id": obj_id})

def _build_admin_users_payload() -> list[dict]:
    """Build merged user list from login accounts and subject profiles (excluding admins)."""
    accounts = list(accounts_collection.find({}, {"password_hash": 0}))

    admin_emails: set[str] = set()
    users_by_email: dict[str, dict] = {}

    for acct in accounts:
        email = (acct.get("email") or "").strip()
        if not email:
            continue

        email_key = email.lower()
        role = _normalize_role(acct.get("role"))
        if role == "admin":
            admin_emails.add(email_key)
            continue

        users_by_email[email_key] = {
            "user_id": str(acct.get("_id")) if acct.get("_id") else None,
            "name": acct.get("name", ""),
            "email": email,
            "role": role,
        }

    # Include legacy profile-only users (created via /register-user) if no account exists yet.
    for profile in users_collection.find({}, {"_id": 1, "name": 1, "email": 1}):
        email = (profile.get("email") or "").strip()
        if not email:
            continue

        email_key = email.lower()
        if email_key in admin_emails:
            continue

        existing = users_by_email.get(email_key)
        if existing:
            if not existing.get("name"):
                existing["name"] = profile.get("name", "")
            continue

        users_by_email[email_key] = {
            "user_id": str(profile.get("_id")) if profile.get("_id") else None,
            "name": profile.get("name", ""),
            "email": email,
            "role": "user",
        }

    users_payload = []
    for user in sorted(users_by_email.values(), key=lambda item: (item.get("email") or "").lower()):
        email = user.get("email", "")
        subject = users_collection.find_one({"email": email}, {"_id": 0}) or {}
        user_results = _list_results_for_email(email, include_details=False)

        users_payload.append({
            **user,
            "subject": subject,
            "results": user_results,
            "result_count": len(user_results),
        })

    return users_payload

async def _get_current_user(token: str = Depends(oauth2_scheme)):
    """Returns account dict or None if unauthenticated."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            return None

        acct = accounts_collection.find_one({"email": email}, {"password_hash": 0})
        if not acct:
            return None

        return {
            "account_id": str(acct.get("_id")) if acct.get("_id") else None,
            "name": acct.get("name", ""),
            "email": acct.get("email", ""),
            "role": _normalize_role(acct.get("role")),
        }
    except JWTError:
        return None

async def _require_auth(token: str = Depends(oauth2_scheme)):
    user = await _get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

async def _require_admin(current_user=Depends(_require_auth)):
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ── Schemas ────────────────────────────────────────────────────────────────────
class UserRegistration(BaseModel):
    name:   str
    email:  str
    age:    Optional[str] = ""
    gender: Optional[str] = ""

class AccountSignup(BaseModel):
    name:     str
    email:    str
    password: str

class AccountLogin(BaseModel):
    email:    str
    password: str

class AdminAccountRegister(BaseModel):
    email:    str
    password: str
    name:     Optional[str] = None


# ── Routes: Health ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "API is running"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


# ── Routes: Authentication ─────────────────────────────────────────────────────
@app.post("/auth/signup")
async def signup(body: AccountSignup):
    if accounts_collection.find_one({"email": body.email}):
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    accounts_collection.insert_one({
        "name":          body.name,
        "email":         body.email,
        "password_hash": _hash_password(body.password),
        "role":          "user",
        "created_at":    datetime.utcnow(),
    })

    # Seed an empty subject-details record so it exists ready to be filled
    users_collection.update_one(
        {"email": body.email},
        {"$setOnInsert": {
            "name": body.name, "email": body.email,
            "age": "", "gender": "",
            "created_at": datetime.utcnow(),
        }},
        upsert=True,
    )

    token = _create_token(body.email, role="user")
    return {
        "token": token,
        "user": {"name": body.name, "email": body.email, "role": "user"},
    }


@app.post("/auth/login")
async def login(body: AccountLogin):
    acct = accounts_collection.find_one({"email": body.email})
    if not acct or not _verify_password(body.password, acct.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    role = _normalize_role(acct.get("role"))
    token = _create_token(body.email, role=role)
    return {
        "token": token,
        "user": {
            "name": acct.get("name", ""),
            "email": acct["email"],
            "role": role,
        },
    }


@app.post("/api/admin/register")
async def admin_register(body: AdminAccountRegister):
    if accounts_collection.find_one({"email": body.email}):
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    admin_name = (body.name or body.email.split("@")[0]).strip() or "admin"
    accounts_collection.insert_one({
        "name":          admin_name,
        "email":         body.email,
        "password_hash": _hash_password(body.password),
        "role":          "admin",
        "created_at":    datetime.utcnow(),
    })

    token = _create_token(body.email, role="admin")
    return {
        "token": token,
        "admin": {"name": admin_name, "email": body.email, "role": "admin"},
    }


@app.post("/api/admin/login")
async def admin_login(body: AccountLogin):
    acct = accounts_collection.find_one({"email": body.email})
    is_valid = acct and _verify_password(body.password, acct.get("password_hash", ""))
    is_admin = bool(acct) and _normalize_role(acct.get("role")) == "admin"
    if not is_valid or not is_admin:
        raise HTTPException(status_code=401, detail="Invalid admin email or password")

    token = _create_token(body.email, role="admin")
    return {
        "token": token,
        "admin": {
            "name": acct.get("name", ""),
            "email": acct["email"],
            "role": "admin",
        },
    }


@app.get("/me")
async def get_me(current_user=Depends(_require_auth)):
    return current_user


# ── Routes: Subject Registration ───────────────────────────────────────────────
@app.post("/register-user")
async def register_user(user: UserRegistration):
    user_doc = user.dict()
    user_doc["created_at"] = datetime.utcnow()

    result = users_collection.update_one(
        {"email": user_doc["email"]},
        {"$set": user_doc},
        upsert=True,
    )

    user_id = str(result.upserted_id) if result.upserted_id else None
    return {"message": "User registered successfully", "user_id": user_id, "email": user_doc["email"]}


# ── Routes: Analysis Results ───────────────────────────────────────────────────
@app.get("/my-results")
async def get_my_results(current_user=Depends(_require_auth)):
    docs = _list_results_for_email(current_user["email"], include_details=False)
    return {"results": docs}


@app.get("/my-results/full/{result_index}")
async def get_my_result_detail(result_index: int, current_user=Depends(_require_auth)):
    """Get a single full result (including timeline + latency) by position (0-based)."""
    docs = list(
        results_collection.find({"email": current_user["email"]})
        .sort("created_at", -1)
        .skip(result_index)
        .limit(1)
    )
    if not docs:
        raise HTTPException(status_code=404, detail="Result not found")
    return _serialize_result(docs[0], include_details=True)


@app.get("/api/admin/users")
async def get_admin_users(_admin_user=Depends(_require_admin)):
    """Return all non-admin users for admin dashboard consumption."""
    users_payload = _build_admin_users_payload()
    return {"users": users_payload, "count": len(users_payload)}


@app.get("/api/admin/users/list")
async def get_admin_users_list(_admin_user=Depends(_require_admin)):
    """Legacy list-only response for older clients."""
    return _build_admin_users_payload()


@app.get("/api/results/user/{user_id}")
async def get_results_for_user(user_id: str, current_user=Depends(_require_auth)):
    """Get summarized results for a user. Allowed for owner or admin."""
    user_email = _resolve_user_email(user_id)
    if not user_email:
        raise HTTPException(status_code=404, detail="User not found")

    _require_owner_or_admin(current_user, user_email)
    return {"results": _list_results_for_email(user_email, include_details=False)}


@app.get("/api/results/detail/{result_id}")
async def get_result_detail(result_id: str, current_user=Depends(_require_auth)):
    """Get a specific full result by result id. Allowed for owner or admin."""
    doc = _find_result_by_id(result_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found")

    _require_owner_or_admin(current_user, doc.get("email", ""))
    return _serialize_result(doc, include_details=True)


@app.get("/api/results/{identifier}")
async def get_results_or_detail(identifier: str, current_user=Depends(_require_auth)):
    """
    Compatibility route:
    - If `identifier` is a valid result id, return one detailed result.
    - Otherwise treat `identifier` as a user id/email and return that user's results.
    """
    doc = _find_result_by_id(identifier)
    if doc:
        _require_owner_or_admin(current_user, doc.get("email", ""))
        return _serialize_result(doc, include_details=True)

    user_email = _resolve_user_email(identifier)
    if not user_email:
        raise HTTPException(status_code=404, detail="Result or user not found")

    _require_owner_or_admin(current_user, user_email)
    return {"results": _list_results_for_email(user_email, include_details=False)}


@app.get("/users")
async def get_all_users(_admin_user=Depends(_require_admin)):
    """Return all registered accounts with their subject details and result counts."""
    users_payload = _build_admin_users_payload()
    return {"users": users_payload}


@app.get("/users/{email}/results")
async def get_user_results(email: str, current_user=Depends(_require_auth)):
    _require_owner_or_admin(current_user, email)
    docs = _list_results_for_email(email, include_details=True)
    return {"results": docs}


# ── Routes: Video Analysis ─────────────────────────────────────────────────────
@app.post("/analyze-video")
async def analyze_video(
    video:      UploadFile       = File(..., description="Video file (mp4, avi, mov, …)"),
    user_email: Optional[str]    = Form(None),
    file_name:  Optional[str]    = Form(None),
):
    """
    Runs the full emotion-suppression pipeline.
    If `user_email` is provided the result is also saved to the `results` collection.

    Response shape
    --------------
    {
      suppression_score, normalized_score, level,
      dominant_emotion, suppressed_emotion,
      timeline, latency_events, files_processed
    }
    """
    original_ext = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=original_ext)
    try:
        tmp.write(await video.read())
        tmp.close()
        result = run_video_pipeline(tmp.name)

        # Persist result to MongoDB when a user is identified
        if user_email:
            results_collection.insert_one({
                "email":              user_email,
                "file_name":          file_name or video.filename or "unknown",
                "suppression_score":  result.get("suppression_score"),
                "normalized_score":   result.get("normalized_score"),
                "level":              result.get("level"),
                "dominant_emotion":   result.get("dominant_emotion"),
                "suppressed_emotion": result.get("suppressed_emotion"),
                "timeline":           result.get("timeline", []),
                "latency_events":     result.get("latency_events", []),
                "files_processed":    result.get("files_processed", 1),
                "created_at":         datetime.utcnow(),
            })

        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ── Dev entry point ────────────────────────────────────────────────────────────
# Run with:  uvicorn main:app --reload --port 8000
