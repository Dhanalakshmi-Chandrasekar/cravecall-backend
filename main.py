from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone, date, timedelta
import os
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import shutil

# ------------------------
# Load environment
# ------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "restaurant")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "orders")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is missing in environment variables")

# ------------------------
# App init
# ------------------------
app = FastAPI(title="Restaurant Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "https://black-plant-05c67ce0f.2.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Static uploads (for logos)
# ------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# If you deploy, set BACKEND_PUBLIC_URL to your domain
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "https://cravecallcateringbk-hrgjcyd3aeaxc3dz.canadacentral-01.azurewebsites.net")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

orders_col = db[COLLECTION_NAME]
users_col = db["users"]
settings_col = db["restaurant_settings"]

# ------------------------
# Helpers
# ------------------------
CLOSED_STATUSES = {"delivered", "cancelled", "completed"}

def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def parse_yyyy_mm_dd(value: str | None) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None

def today_yyyy_mm_dd_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ------------------------
# SETTINGS DEFAULTS + MERGE
# ------------------------
def default_settings_doc() -> Dict[str, Any]:
    return {
        "restaurant_id": "default",

        "restaurant_name": "Savory Bites Catering",
        "business_phone": "(555) 000-0000",
        "business_email": "contact@savorybites.com",
        "business_address": "123 Main Street, Suite 100, City, State 12345",

        "branding": {
            "app_name": "Savory Bites",
            "tagline": "Catering Manager",
            "logo_url": ""
        },

        "notifications": {
            "email_new_orders": True,
            "sms_urgent_orders": True,
            "weekly_revenue_reports": False,
            "customer_feedback": True
        },

        "billing": {
            "plan_name": "Premium Plan",
            "status": "active",
            "price_monthly": 99
        },

        "updated_at": datetime.now(timezone.utc).isoformat()
    }

def merge_with_defaults(existing: Dict[str, Any]) -> Dict[str, Any]:
    base = default_settings_doc()
    for k, v in existing.items():
        base[k] = v

    base["branding"] = {**default_settings_doc()["branding"], **(existing.get("branding") or {})}
    base["notifications"] = {**default_settings_doc()["notifications"], **(existing.get("notifications") or {})}
    base["billing"] = {**default_settings_doc()["billing"], **(existing.get("billing") or {})}

    return base

# ------------------------
# Pydantic models
# ------------------------
class BrandingUpdate(BaseModel):
    app_name: Optional[str] = None
    tagline: Optional[str] = None
    logo_url: Optional[str] = None

class NotificationsUpdate(BaseModel):
    email_new_orders: Optional[bool] = None
    sms_urgent_orders: Optional[bool] = None
    weekly_revenue_reports: Optional[bool] = None
    customer_feedback: Optional[bool] = None

class BillingUpdate(BaseModel):
    plan_name: Optional[str] = None
    status: Optional[str] = None
    price_monthly: Optional[float] = None

class SettingsUpdate(BaseModel):
    restaurant_name: Optional[str] = None
    business_phone: Optional[str] = None
    business_email: Optional[str] = None
    business_address: Optional[str] = None
    branding: Optional[BrandingUpdate] = None
    notifications: Optional[NotificationsUpdate] = None
    billing: Optional[BillingUpdate] = None

class AccountUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None

# ------------------------
# Logo Upload Endpoint ✅
# ------------------------
@app.post("/settings/branding/logo")
def upload_branding_logo(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
        # fallback safe extension
        ext = ".png"

    filename = f"logo_{uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    # Save file
    with open(path, "wb") as f:
        f.write(file.file.read())

    logo_url = f"{BACKEND_PUBLIC_URL}/uploads/{filename}"

    # Persist directly (so sidebar can load it)
    settings_col.update_one(
        {"restaurant_id": "default"},
        {
            "$set": {
                "branding.logo_url": logo_url,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        },
        upsert=True
    )

    return {"success": True, "logo_url": logo_url}

# ------------------------
# SEED ORDERS
# ------------------------
@app.post("/seed/orders")
def seed_orders(clear_existing: bool = False):
    if clear_existing:
        orders_col.delete_many({})

    now_iso = datetime.now(timezone.utc).isoformat()

    sample_orders = [
        {
            "order_id": str(uuid4()),
            "status": "confirmed",
            "currency": "USD",
            "customer_name": "John Smith",
            "phone": "+1-555-234-5678",
            "guest_count": 35,
            "delivery_type": "delivery",
            "address": "123 Main Street, Dallas, TX",
            "event_date": "2026-01-15",
            "event_time": "6:00 PM",
            "items": [
                {"item_name": "Chicken Biryani Tray", "type": "tray", "quantity": 4, "unit_price_usd": 260, "total_price_usd": 1040}
            ],
            "grand_total_usd": 1040,
            "special_instructions": "Extra spicy",
            "created_at": now_iso
        },
        {
            "order_id": str(uuid4()),
            "status": "preparing",
            "currency": "USD",
            "customer_name": "Emily Davis",
            "phone": "+1-555-987-1234",
            "guest_count": 20,
            "delivery_type": "pickup",
            "address": None,
            "event_date": "2026-01-14",
            "event_time": "12:30 PM",
            "items": [
                {"item_name": "Veg Box Meal", "type": "box", "quantity": 20, "unit_price_usd": 12, "total_price_usd": 240}
            ],
            "grand_total_usd": 240,
            "special_instructions": "",
            "created_at": now_iso
        },
    ]

    orders_col.insert_many(sample_orders)
    return {"success": True, "inserted": len(sample_orders), "cleared_existing": clear_existing}

# ------------------------
# Orders
# ------------------------
@app.get("/orders")
def get_orders(status: str = None):
    query = {}
    if status:
        query["status"] = status
    return list(orders_col.find(query, {"_id": 0}))

@app.get("/orders/{order_id}")
def get_order(order_id: str):
    order = orders_col.find_one({"order_id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@app.put("/orders/{order_id}/status")
def update_order_status(
    order_id: str,
    status: str = Query(enum=["confirmed", "preparing", "ready", "delivered", "completed", "cancelled"])
):
    result = orders_col.update_one(
        {"order_id": order_id},
        {"$set": {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"success": True, "order_id": order_id, "status": status}

@app.put("/orders/{order_id}")
def update_order_details(order_id: str, updates: dict):
    allowed_fields = {"customer_name", "phone", "address", "event_date", "event_time", "special_instructions", "delivery_type"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed_fields}

    if not safe_updates:
        raise HTTPException(status_code=400, detail="No valid fields provided for update")

    order = orders_col.find_one({"order_id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order.get("status") in ["preparing", "ready", "delivered", "completed"]:
        raise HTTPException(status_code=403, detail="Order cannot be edited after preparation started")

    safe_updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    orders_col.update_one({"order_id": order_id}, {"$set": safe_updates})

    return {"success": True, "updated_fields": list(safe_updates.keys())}

# ------------------------
# Dashboard Summary
# ------------------------
@app.get("/dashboard/summary")
def dashboard_summary(
    day: str | None = Query(default=None, description="YYYY-MM-DD. If not provided, uses today's date in UTC."),
    recent_limit: int = Query(default=5, ge=1, le=50)
):
    day_str = day or today_yyyy_mm_dd_utc()
    day_date = parse_yyyy_mm_dd(day_str)
    if not day_date:
        raise HTTPException(status_code=400, detail="Invalid day format. Use YYYY-MM-DD")

    projection = {
        "_id": 0,
        "order_id": 1,
        "customer_name": 1,
        "phone": 1,
        "guest_count": 1,
        "grand_total_usd": 1,
        "status": 1,
        "created_at": 1,
        "event_date": 1,
        "event_time": 1
    }

    all_orders = list(orders_col.find({}, projection))

    total_orders_today = 0
    total_revenue_today = 0.0
    upcoming_events = 0
    pending_orders = 0

    for o in all_orders:
        created_dt = parse_iso_datetime(o.get("created_at"))
        if created_dt and created_dt.date() == day_date:
            total_orders_today += 1
            total_revenue_today += float(o.get("grand_total_usd") or 0)

        ev_date = parse_yyyy_mm_dd(o.get("event_date"))
        if ev_date and ev_date >= day_date:
            upcoming_events += 1

        status = str(o.get("status") or "").lower()
        if status not in CLOSED_STATUSES:
            pending_orders += 1

    def safe_dt_sort(v: str | None) -> float:
        dt = parse_iso_datetime(v)
        return dt.timestamp() if dt else 0.0

    recent_orders = sorted(all_orders, key=lambda o: safe_dt_sort(o.get("created_at")), reverse=True)[:recent_limit]

    for idx, o in enumerate(recent_orders):
        if not o.get("order_id"):
            o["order_id"] = f"TEMP-{idx}"

    return {
        "day": day_str,
        "total_orders_today": total_orders_today,
        "total_revenue_today": round(total_revenue_today, 2),
        "upcoming_events": upcoming_events,
        "pending_orders": pending_orders,
        "recent_orders": recent_orders
    }

# ------------------------
# Customers Summary
# ------------------------
@app.get("/customers/summary")
def customers_summary():
    projection = {"_id": 0, "customer_name": 1, "phone": 1, "grand_total_usd": 1, "created_at": 1, "order_id": 1}
    orders = list(orders_col.find({}, projection))

    customers: Dict[str, Any] = {}

    for o in orders:
        name = (o.get("customer_name") or "").strip() or "Unknown"
        phone = (o.get("phone") or "").strip() or ""
        key = phone if phone else name.lower()
        total = float(o.get("grand_total_usd") or 0)

        if key not in customers:
            customers[key] = {"id": key, "name": name, "phone": phone if phone else "-", "orderCount": 0, "totalSpend": 0.0}

        customers[key]["orderCount"] += 1
        customers[key]["totalSpend"] += total

    customer_list = list(customers.values())
    customer_list.sort(key=lambda c: c["totalSpend"], reverse=True)

    total_customers = len(customer_list)
    total_spend = sum(c["totalSpend"] for c in customer_list)
    avg_spend = (total_spend / total_customers) if total_customers else 0.0
    top_customer = customer_list[0] if total_customers else None

    return {
        "total_customers": total_customers,
        "total_spend": round(total_spend, 2),
        "avg_spend_per_customer": round(avg_spend, 2),
        "top_customer": top_customer,
        "customers": customer_list
    }

# ------------------------
# Revenue Summary
# ------------------------
@app.get("/revenue/summary")
def revenue_summary(days: int = Query(default=8, ge=1, le=60)):
    projection = {"_id": 0, "order_id": 1, "created_at": 1, "grand_total_usd": 1, "delivery_type": 1, "items": 1}
    orders = list(orders_col.find({}, projection))

    now = datetime.now(timezone.utc)
    start_day = (now - timedelta(days=days - 1)).date()

    daily_map: Dict[str, float] = {}
    for i in range(days):
        d = (start_day + timedelta(days=i)).strftime("%Y-%m-%d")
        daily_map[d] = 0.0

    total_revenue = 0.0
    order_count = 0
    highest_order = 0.0
    delivery_revenue = 0.0
    pickup_revenue = 0.0
    item_map: Dict[str, Any] = {}

    for o in orders:
        total = float(o.get("grand_total_usd") or 0)
        total_revenue += total
        order_count += 1
        highest_order = max(highest_order, total)

        dtype = str(o.get("delivery_type") or "").lower()
        if dtype == "delivery":
            delivery_revenue += total
        elif dtype == "pickup":
            pickup_revenue += total

        created_dt = parse_iso_datetime(o.get("created_at"))
        if created_dt:
            dkey = created_dt.date().strftime("%Y-%m-%d")
            if dkey in daily_map:
                daily_map[dkey] += total

        for it in (o.get("items") or []):
            name = (it.get("item_name") or "").strip()
            if not name:
                continue
            rev = float(it.get("total_price_usd") or 0)
            if name not in item_map:
                item_map[name] = {"name": name, "revenue": 0.0, "orders": 0}
            item_map[name]["revenue"] += rev
            item_map[name]["orders"] += 1

    avg_order_value = (total_revenue / order_count) if order_count else 0.0

    revenue_over_time = [{"date": d, "revenue": round(v, 2)} for d, v in daily_map.items()]

    top_items = sorted(item_map.values(), key=lambda x: x["revenue"], reverse=True)[:5]
    for x in top_items:
        x["revenue"] = round(float(x["revenue"]), 2)

    return {
        "days": days,
        "total_revenue": round(total_revenue, 2),
        "avg_order_value": round(avg_order_value, 2),
        "highest_order": round(highest_order, 2),
        "revenue_over_time": revenue_over_time,
        "revenue_by_delivery_type": {
            "delivery": round(delivery_revenue, 2),
            "pickup": round(pickup_revenue, 2),
        },
        "top_items": top_items,
        "date_range": {
            "start": start_day.strftime("%Y-%m-%d"),
            "end": now.date().strftime("%Y-%m-%d"),
        },
    }

# ------------------------
# Settings GET + PUT
# ------------------------
@app.get("/settings")
def get_settings():
    existing = settings_col.find_one({"restaurant_id": "default"}, {"_id": 0}) or {}
    if not existing:
        doc = default_settings_doc()
        settings_col.insert_one(doc)
        return doc
    return merge_with_defaults(existing)

@app.put("/settings")
def update_settings(payload: SettingsUpdate):
    data = payload.model_dump(exclude_none=True)

    update_doc: Dict[str, Any] = {}
    branding = data.pop("branding", None)
    notifications = data.pop("notifications", None)
    billing = data.pop("billing", None)

    for k, v in data.items():
        update_doc[k] = v

    if branding:
        for k, v in branding.items():
            update_doc[f"branding.{k}"] = v

    if notifications:
        for k, v in notifications.items():
            update_doc[f"notifications.{k}"] = v

    if billing:
        for k, v in billing.items():
            update_doc[f"billing.{k}"] = v

    if not update_doc:
        raise HTTPException(status_code=400, detail="No fields to update")

    update_doc["updated_at"] = datetime.now(timezone.utc).isoformat()

    settings_col.update_one(
        {"restaurant_id": "default"},
        {"$set": update_doc},
        upsert=True
    )

    return {"success": True}

# ------------------------
# Users /users/me
# ------------------------
@app.get("/users/me")
def get_me():
    user = users_col.find_one({"user_id": "default"}, {"_id": 0})
    if not user:
        user = {
            "user_id": "default",
            "name": "John Manager",
            "email": "manager@savorybites.com",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        users_col.insert_one(user)
    return user

@app.put("/users/me")
def update_me(payload: AccountUpdate):
    update_doc = payload.model_dump(exclude_none=True)
    if not update_doc:
        raise HTTPException(status_code=400, detail="No fields to update")

    update_doc["updated_at"] = datetime.now(timezone.utc).isoformat()

    users_col.update_one(
        {"user_id": "default"},
        {"$set": update_doc},
        upsert=True
    )

    return {"success": True}


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/settings/branding/logo")
async def upload_logo(file: UploadFile = File(...)):
    # Basic validation
    allowed = {"image/png", "image/jpeg", "image/webp", "image/svg+xml"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    ext = os.path.splitext(file.filename or "")[1] or ".png"
    fname = f"logo-{uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)

    with open(fpath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Serve URL (same backend)
    logo_url = f"http://127.0.0.1:9002/uploads/{fname}"

    # optionally persist in settings immediately
    settings_col.update_one(
        {"restaurant_id": "default"},
        {"$set": {"branding.logo_url": logo_url, "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True
    )

    return {"success": True, "logo_url": logo_url}

@app.get("/uploads/{filename}")
def get_uploaded_file(filename: str):
    fpath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fpath)


