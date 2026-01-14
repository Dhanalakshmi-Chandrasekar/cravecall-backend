import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from pymongo import MongoClient
from jose import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# =========================================================
# LOGGING (Azure Log Stream)
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("restaurant-api")

# =========================================================
# ENV
# =========================================================
load_dotenv()
logger.info("Environment variables loaded")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "restaurant_db")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGO = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "1440"))

if not MONGO_URI:
    logger.error("MONGO_URI not set")
    raise RuntimeError("MONGO_URI missing")

# =========================================================
# APP
# =========================================================
app = FastAPI(title="Restaurant Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DB
# =========================================================
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    users_col = db["users"]
    logger.info("MongoDB connected")
except Exception:
    logger.exception("MongoDB connection failed")
    raise

# =========================================================
# SECURITY
# =========================================================
ph = PasswordHasher()


def hash_password(password: str) -> str:
    return ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def create_access_token(payload: dict) -> str:
    data = payload.copy()
    data["exp"] = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MIN)
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGO)

# =========================================================
# SCHEMAS
# =========================================================
class RegisterRequest(BaseModel):
    name: str | None = None
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    user: dict

# =========================================================
# AUTH ROUTES
# =========================================================
@app.post("/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest):
    logger.info(f"Register attempt: {payload.email}")

    email = payload.email.lower().strip()

    if users_col.find_one({"email": email}):
        logger.warning(f"Register failed – email exists: {email}")
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    doc = {
        "user_id": user_id,
        "name": (payload.name or "User").strip(),
        "email": email,
        "password_hash": hash_password(payload.password),
        "created_at": now,
        "updated_at": now,
    }

    users_col.insert_one(doc)
    logger.info(f"User registered: {email}")

    token = create_access_token({"sub": user_id, "email": email})

    return {
        "access_token": token,
        "user": {
            "user_id": user_id,
            "name": doc["name"],
            "email": email,
        },
    }


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    logger.info(f"Login attempt: {payload.email}")

    email = payload.email.lower().strip()
    user = users_col.find_one({"email": email})

    if not user:
        logger.warning(f"Login failed – user not found: {email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(payload.password, user.get("password_hash", "")):
        logger.warning(f"Login failed – wrong password: {email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user["user_id"], "email": email})
    logger.info(f"Login success: {email}")

    return {
        "access_token": token,
        "user": {
            "user_id": user["user_id"],
            "name": user.get("name", ""),
            "email": email,
        },
    }

# =========================================================
# FILE UPLOAD (example)
# =========================================================
@app.post("/settings/branding/logo")
async def upload_logo(file: UploadFile = File(...)):
    logger.info(f"Logo upload: {file.filename} | {file.content_type}")
    return {"success": True, "filename": file.filename}

# =========================================================
# GLOBAL ERROR HANDLER (CRITICAL FOR AZURE)
# =========================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error | {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}
