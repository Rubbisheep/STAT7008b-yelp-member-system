import os
import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .text_mining import classify_user_segment, extract_keywords, predict_mbti


def _utcnow() -> str:
    return datetime.utcnow().isoformat(sep=" ", timespec="seconds")


class MusicAppSystem:
    """
    Lightweight service layer around SQLite for the music membership system.
    Methods cover the assignment-required operations end to end.
    """

    def __init__(self, db_path: str = "data/music.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- User functions ---
    def register_user(
        self,
        email: Optional[str],
        phone: Optional[str],
        nickname: str,
        gender: Optional[str] = None,
        birth_year: Optional[int] = None,
        region: Optional[str] = None,
        register_source: Optional[str] = None,
        invited_by_user_id: Optional[int] = None,
        user_id: Optional[int] = None,
        register_time: Optional[str] = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (
                    user_id, email, phone, nickname, gender, birth_year, region,
                    register_source, register_time, invited_by_user_id, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    user_id,
                    email,
                    phone,
                    nickname,
                    gender,
                    birth_year,
                    region,
                    register_source,
                    register_time or _utcnow(),
                    invited_by_user_id,
                    _utcnow(),
                    _utcnow(),
                ),
            )
            return cur.lastrowid if user_id is None else user_id

    def login_user(self, user_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            return cur.fetchone() is not None

    def update_user_profile(self, user_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join([f"{k} = ?" for k in fields])
        params = list(fields.values()) + [user_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE users SET {cols}, updated_at = ? WHERE user_id = ?", params + [_utcnow(), user_id])

    def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    # --- Membership ---
    def subscribe_membership(
        self,
        user_id: int,
        plan_id: Optional[int],
        duration_days: int,
        auto_renew: bool = False,
    ) -> int:
        start = datetime.utcnow()
        end = start + timedelta(days=duration_days)
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO membership_subscriptions (
                    user_id, plan_id, status, start_at, end_at, auto_renew,
                    created_at, updated_at
                ) VALUES (?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    plan_id,
                    start.isoformat(sep=" ", timespec="seconds"),
                    end.isoformat(sep=" ", timespec="seconds"),
                    int(auto_renew),
                    _utcnow(),
                    _utcnow(),
                ),
            )
            return cur.lastrowid

    def cancel_membership(self, sub_id: int, reason: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE membership_subscriptions
                SET status = 'cancelled', cancel_reason = ?, updated_at = ?
                WHERE sub_id = ?
                """,
                (reason, _utcnow(), sub_id),
            )

    def get_membership_status(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                """
                SELECT * FROM membership_subscriptions
                WHERE user_id = ?
                ORDER BY start_at DESC
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # --- Event logging ---
    def log_listening_session(
        self,
        user_id: int,
        play_time: str,
        duration_sec: int,
        genre: Optional[str] = None,
        track_id: Optional[int] = None,
        is_skipped: bool = False,
        device_type: str = "mobile",
        from_recommend: bool = False,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO listening_logs (
                    user_id, play_time, duration_sec, track_id, genre,
                    is_skipped, device_type, from_recommend, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    play_time,
                    duration_sec,
                    track_id,
                    genre,
                    int(is_skipped),
                    device_type,
                    int(from_recommend),
                    _utcnow(),
                ),
            )
            return cur.lastrowid

    def log_feedback(
        self,
        user_id: int,
        feedback_time: str,
        rating: int,
        comment_text: str,
        channel: str = "in_app",
        feedback_type: str = "overall",
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedbacks (
                    user_id, feedback_time, rating, channel, feedback_type,
                    comment_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    feedback_time,
                    rating,
                    channel,
                    feedback_type,
                    comment_text,
                    _utcnow(),
                ),
            )
            return cur.lastrowid

    # --- Analytics ---
    def get_user_activity_stats(self, user_id: int) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS play_count,
                    IFNULL(SUM(duration_sec), 0) AS total_duration,
                    MAX(play_time) AS last_play_time
                FROM listening_logs
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            return dict(row) if row else {}

    def get_global_trends(self) -> Dict[str, List[Tuple[str, int]]]:
        with self._conn() as conn:
            regs = conn.execute(
                """
                SELECT substr(register_time, 1, 10) AS day, COUNT(*) AS cnt
                FROM users
                GROUP BY day
                ORDER BY day
                """
            ).fetchall()
            dau = conn.execute(
                """
                SELECT substr(play_time, 1, 10) AS day, COUNT(DISTINCT user_id) AS cnt
                FROM listening_logs
                GROUP BY day
                ORDER BY day
                """
            ).fetchall()
            return {
                "register_trend": [(r["day"], r["cnt"]) for r in regs],
                "dau_trend": [(r["day"], r["cnt"]) for r in dau],
            }

    def compute_churn_risk(self, user_id: int) -> str:
        stats = self.get_user_activity_stats(user_id)
        last_str = stats.get("last_play_time")
        if not last_str:
            return "high"
        last = datetime.fromisoformat(last_str)
        delta = datetime.utcnow() - last
        if delta.days >= 30:
            return "high"
        if delta.days >= 14:
            return "medium"
        return "low"

    # --- Text mining and segmentation ---
    def update_user_profile_from_text(self, user_id: int) -> None:
        texts = []
        with self._conn() as conn:
            pref = conn.execute(
                "SELECT extra_info FROM user_preferences WHERE user_id = ?", (user_id,)
            ).fetchone()
            if pref and pref["extra_info"]:
                texts.append(pref["extra_info"])
            rows = conn.execute(
                "SELECT comment_text FROM feedbacks WHERE user_id = ?", (user_id,)
            ).fetchall()
        texts.extend([r["comment_text"] for r in rows if r["comment_text"]])
        combined = " ".join(texts)
        keywords = extract_keywords(combined, top_k=8)
        mbti = predict_mbti(combined)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, mbti_guess, keywords_summary, last_profile_update
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    mbti_guess = excluded.mbti_guess,
                    keywords_summary = excluded.keywords_summary,
                    last_profile_update = excluded.last_profile_update
                """,
                (
                    user_id,
                    mbti,
                    ",".join(keywords),
                    _utcnow(),
                ),
            )

    def update_user_profile_from_behavior(self, user_id: int) -> None:
        with self._conn() as conn:
            top_genre = conn.execute(
                """
                SELECT genre, COUNT(*) AS c
                FROM listening_logs
                WHERE user_id = ? AND genre IS NOT NULL
                GROUP BY genre
                ORDER BY c DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            active = conn.execute(
                """
                SELECT
                  CASE
                    WHEN CAST(substr(play_time, 12, 2) AS INTEGER) BETWEEN 0 AND 6 THEN 'late_night'
                    WHEN CAST(substr(play_time, 12, 2) AS INTEGER) BETWEEN 7 AND 12 THEN 'morning'
                    WHEN CAST(substr(play_time, 12, 2) AS INTEGER) BETWEEN 13 AND 18 THEN 'afternoon'
                    ELSE 'evening'
                  END AS bucket,
                  COUNT(*) AS c
                FROM listening_logs
                WHERE user_id = ?
                GROUP BY bucket
                ORDER BY c DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        main_interest = top_genre["genre"] if top_genre else None
        active_pattern = active["bucket"] if active else None
        churn = self.compute_churn_risk(user_id)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, main_interest, active_time_pattern,
                    churn_risk_level, last_profile_update
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    main_interest = excluded.main_interest,
                    active_time_pattern = excluded.active_time_pattern,
                    churn_risk_level = excluded.churn_risk_level,
                    last_profile_update = excluded.last_profile_update
                """,
                (
                    user_id,
                    main_interest,
                    active_pattern,
                    churn,
                    _utcnow(),
                ),
            )

    def classify_user_segment(self, user_id: int) -> Optional[int]:
        with self._conn() as conn:
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) AS play_count,
                    IFNULL(SUM(duration_sec), 0) AS total_duration,
                    SUM(CASE WHEN is_skipped = 1 THEN 1 ELSE 0 END) AS skips
                FROM listening_logs
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            feedback_avg = conn.execute(
                "SELECT AVG(rating) AS avg_rating FROM feedbacks WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            features = {
                "play_count": stats["play_count"] or 0,
                "total_duration": stats["total_duration"] or 0,
                "skips": stats["skips"] or 0,
                "avg_rating": feedback_avg["avg_rating"] or 0,
            }
        segment_name = classify_user_segment(features)
        segment_id = self._ensure_segment(segment_name)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_segment_membership (
                    user_id, segment_id, assigned_by, assigned_at
                ) VALUES (?, ?, 'classifier', ?)
                ON CONFLICT(user_id, segment_id) DO NOTHING
                """,
                (user_id, segment_id, _utcnow()),
            )
        return segment_id

    def _ensure_segment(self, segment_name: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT segment_id FROM user_segments WHERE segment_name = ?",
                (segment_name,),
            ).fetchone()
            if row:
                return row["segment_id"]
            cur = conn.execute(
                "INSERT INTO user_segments (segment_name, description) VALUES (?, ?)",
                (segment_name, f"Auto-created segment {segment_name}"),
            )
            return cur.lastrowid
