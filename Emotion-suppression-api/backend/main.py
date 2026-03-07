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
users_collection    = db["users"]       # subject details (name/email/phone/age/gender)
accounts_collection = db["accounts"]    # login credentials
results_collection  = db["results"]     # per-user analysis results

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://suppression.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth helpers ───────────────────────────────────────────────────────────────
def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def _hash_password(password: str) -> str:
    return pwd_context.hash(password)

def _create_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_TTL_MINUTES)
    return jwt.encode({"sub": email, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def _get_current_user(token: str = Depends(oauth2_scheme)):
    """Returns account dict or None if unauthenticated."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            return None
        return accounts_collection.find_one({"email": email}, {"_id": 0, "password_hash": 0})
    except JWTError:
        return None

async def _require_auth(token: str = Depends(oauth2_scheme)):
    user = await _get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

# ── Schemas ────────────────────────────────────────────────────────────────────
class UserRegistration(BaseModel):
    name:   str
    email:  str
    phone:  Optional[str] = None
    age:    Optional[str] = ""
    gender: Optional[str] = ""

class AccountSignup(BaseModel):
    name:     str
    email:    str
    password: str

class AccountLogin(BaseModel):
    email:    str
    password: str


# ── Routes: Health ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/health")
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
        "created_at":    datetime.utcnow(),
    })

    # Seed an empty subject-details record so it exists ready to be filled
    users_collection.update_one(
        {"email": body.email},
        {"$setOnInsert": {
            "name": body.name, "email": body.email,
            "phone": "", "age": "", "gender": "",
            "created_at": datetime.utcnow(),
        }},
        upsert=True,
    )

    token = _create_token(body.email)
    return {"token": token, "user": {"name": body.name, "email": body.email}}


@app.post("/auth/login")
async def login(body: AccountLogin):
    acct = accounts_collection.find_one({"email": body.email})
    if not acct or not _verify_password(body.password, acct["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(body.email)
    return {"token": token, "user": {"name": acct["name"], "email": acct["email"]}}


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
    docs = list(results_collection.find(
        {"email": current_user["email"]},
        {"_id": 0, "timeline": 0, "latency_events": 0}
    ).sort("created_at", -1))
    return {"results": docs}


@app.get("/my-results/full/{result_index}")
async def get_my_result_detail(result_index: int, current_user=Depends(_require_auth)):
    """Get a single full result (including timeline + latency) by position (0-based)."""
    doc = list(results_collection.find(
        {"email": current_user["email"]}, {"_id": 0}
    ).sort("created_at", -1).skip(result_index).limit(1))
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found")
    return doc[0]


@app.get("/users")
async def get_all_users(_current_user=Depends(_require_auth)):
    """Return all registered accounts with their subject details and result counts."""
    accounts = list(accounts_collection.find({}, {"_id": 0, "password_hash": 0}))
    enriched = []
    for acct in accounts:
        subject = users_collection.find_one({"email": acct["email"]}, {"_id": 0}) or {}
        user_results = list(results_collection.find(
            {"email": acct["email"]},
            {"_id": 0, "timeline": 0, "latency_events": 0}
        ).sort("created_at", -1))
        enriched.append({
            **acct,
            "subject":      subject,
            "results":      user_results,
            "result_count": len(user_results),
        })
    return {"users": enriched}


@app.get("/users/{email}/results")
async def get_user_results(email: str, _current_user=Depends(_require_auth)):
    docs = list(results_collection.find(
        {"email": email}, {"_id": 0}
    ).sort("created_at", -1))
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
