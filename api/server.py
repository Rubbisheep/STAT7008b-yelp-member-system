from pathlib import Path
from typing import List, Optional, Dict
import threading
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.music_app_system import MusicAppSystem  # noqa: E402
from scripts import init_db as init_db_script  # noqa: E402
from scripts import simulate_data as simulate_data_script  # noqa: E402
from scripts import generate_dashboard as generate_dashboard_script  # noqa: E402


DB_PATH = ROOT / "data" / "music.db"
os.makedirs(DB_PATH.parent, exist_ok=True)
app = FastAPI(title="Music Membership API")
system = MusicAppSystem(str(DB_PATH))
tasks_status: Dict[str, Dict[str, str]] = {}

# Allow local static page calls (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    nickname: str
    gender: Optional[str] = None
    birth_year: Optional[int] = None
    region: Optional[str] = None
    register_source: Optional[str] = "web"
    fav_genres: Optional[str] = None
    fav_scenes: Optional[str] = None
    extra_info: Optional[str] = None


@app.post("/api/register")
def register_user(req: RegisterRequest):
    user_id = system.register_user(
        email=req.email,
        phone=req.phone,
        nickname=req.nickname,
        gender=req.gender,
        birth_year=req.birth_year,
        region=req.region,
        register_source=req.register_source,
    )
    # upsert preferences if provided
    if req.fav_genres or req.fav_scenes or req.extra_info:
        with system._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (
                    user_id, fav_genres, fav_scenes, extra_info, created_at, updated_at
                ) VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(user_id) DO UPDATE SET
                    fav_genres = excluded.fav_genres,
                    fav_scenes = excluded.fav_scenes,
                    extra_info = excluded.extra_info,
                    updated_at = datetime('now')
                """,
                (user_id, req.fav_genres, req.fav_scenes, req.extra_info),
            )
    return {"user_id": user_id, "message": "registered"}


@app.get("/api/users/search")
def search_users(
    q: str = Query(..., min_length=1, description="search by nickname/email/phone"),
    limit: int = 20,
):
    pattern = f"%{q}%"
    with system._conn() as conn:
        rows = conn.execute(
            """
            SELECT
                u.user_id,
                u.nickname,
                u.email,
                u.phone,
                u.region,
                u.register_time,
                u.register_source,
                u.invited_by_user_id,
                u.status,
                p.fav_genres,
                p.fav_scenes,
                p.extra_info,
                (
                    SELECT status
                    FROM membership_subscriptions ms
                    WHERE ms.user_id = u.user_id
                    ORDER BY start_at DESC
                    LIMIT 1
                ) AS membership_status
            FROM users u
            LEFT JOIN user_preferences p ON p.user_id = u.user_id
            WHERE u.nickname LIKE ? OR u.email LIKE ? OR u.phone LIKE ?
            ORDER BY u.register_time DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, limit),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    info = system.get_user_info(user_id)
    if not info:
        raise HTTPException(status_code=404, detail="User not found")
    # attach preferences & profile
    with system._conn() as conn:
        pref = conn.execute(
            "SELECT fav_genres, fav_scenes, extra_info FROM user_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        profile = conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    info["preferences"] = dict(pref) if pref else None
    info["profile"] = dict(profile) if profile else None
    return info


@app.get("/")
def root():
    return {"message": "Music Membership API", "docs": "/docs"}


# ----- Actions with async-style tracking -----
def _run_task(task_id: str, action: str, func):
    tasks_status[task_id] = {"action": action, "status": "running"}
    try:
        func()
        tasks_status[task_id] = {"action": action, "status": "completed"}
    except Exception as exc:  # noqa: BLE001
        tasks_status[task_id] = {"action": action, "status": "failed", "error": str(exc)}


def _start_task(action: str, func):
    task_id = str(uuid.uuid4())
    tasks_status[task_id] = {"action": action, "status": "queued"}
    thread = threading.Thread(target=_run_task, args=(task_id, action, func), daemon=True)
    thread.start()
    return task_id


@app.get("/api/actions/status")
def action_status(task_id: Optional[str] = None):
    if task_id:
        return tasks_status.get(task_id, {"status": "unknown"})
    return tasks_status


@app.post("/api/actions/init_db")
def action_init_db():
    task_id = _start_task("init_db", init_db_script.init_db)
    return {"status": "queued", "action": "init_db", "task_id": task_id}


@app.post("/api/actions/simulate_data")
def action_simulate_data():
    task_id = _start_task("simulate_data", simulate_data_script.main)
    return {"status": "queued", "action": "simulate_data", "task_id": task_id}


@app.post("/api/actions/generate_dashboard")
def action_generate_dashboard():
    task_id = _start_task("generate_dashboard", generate_dashboard_script.main)
    return {"status": "queued", "action": "generate_dashboard", "task_id": task_id}


# To run: uvicorn api.server:app --reload --port 8001
