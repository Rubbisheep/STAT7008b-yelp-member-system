import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.music_app_system import MusicAppSystem, _utcnow  # noqa: E402


DB_PATH = ROOT / "data" / "music.db"

faker = Faker("en_US")
random.seed(42)
Faker.seed(42)

GENRES = ["pop", "rock", "jazz", "hiphop", "classical", "electronic"]
SCENES = ["study", "workout", "commute", "sleep"]
FEEDBACK_TEXTS = [
    "Recommendations are inaccurate",
    "Playlist feature is handy",
    "Please add more classical music",
    "UI feels great",
    "Good rhythm",
    "Song catalog is rich",
    "Price feels a bit high",
]


def simulate_users(system: MusicAppSystem, n: int = 800) -> list[int]:
    user_ids = []
    now = datetime.utcnow()
    for i in range(n):
        nickname = faker.name()
        reg_time = (now - timedelta(days=random.randint(0, 90), hours=random.randint(0, 23))).replace(
            microsecond=0
        ).isoformat(sep=" ")
        user_id = system.register_user(
            email=faker.email(),
            phone=faker.phone_number(),
            nickname=nickname,
            gender=random.choice(["M", "F", None]),
            birth_year=random.randint(1975, 2010),
            region=faker.state(),
            register_source=random.choice(["ads", "friend", "social_media", "search"]),
            invited_by_user_id=random.choice(user_ids) if user_ids and random.random() < 0.2 else None,
            register_time=reg_time,
        )
        user_ids.append(user_id)
    return user_ids


def simulate_preferences(system: MusicAppSystem, user_ids: list[int]) -> None:
    with system._conn() as conn:
        for uid in user_ids:
            fav_genres = random.sample(GENRES, k=random.randint(1, 3))
            fav_scenes = random.sample(SCENES, k=random.randint(1, 2))
            extra = random.choice(
                [
                    "I enjoy melancholic songs",
                    "I like soothing melodies",
                    "Need strong beats for workouts",
                    "Energizing music for commute",
                    "Looking for sleep aid music",
                ]
            )
            conn.execute(
                """
                INSERT INTO user_preferences (
                    user_id, fav_genres, fav_scenes, extra_info, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    ",".join(fav_genres),
                    ",".join(fav_scenes),
                    extra,
                    _utcnow(),
                    _utcnow(),
                ),
            )


def weighted_choice(genres: list[str]) -> str:
    weights = []
    for g in GENRES:
        weights.append(3 if g in genres else 1)
    return random.choices(GENRES, weights=weights, k=1)[0]


def simulate_listening(system: MusicAppSystem, user_ids: list[int]) -> None:
    now = datetime.utcnow()
    with system._conn() as conn:
        for uid in user_ids:
            pref_row = conn.execute(
                "SELECT fav_genres FROM user_preferences WHERE user_id = ?", (uid,)
            ).fetchone()
            favs = pref_row["fav_genres"].split(",") if pref_row and pref_row["fav_genres"] else []
            sessions = random.randint(5, 200)
            for _ in range(sessions):
                delta_days = random.randint(0, 60)
                play_time = (now - timedelta(days=delta_days, hours=random.randint(0, 23))).replace(
                    microsecond=0
                )
                genre = weighted_choice(favs) if favs else random.choice(GENRES)
                conn.execute(
                    """
                    INSERT INTO listening_logs (
                        user_id, play_time, duration_sec, genre, is_skipped,
                        device_type, from_recommend, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uid,
                        play_time.isoformat(sep=" "),
                        random.randint(30, 300),
                        genre,
                        int(random.random() < 0.05),
                        random.choice(["mobile", "web"]),
                        int(random.random() < 0.5),
                        _utcnow(),
                    ),
                )


def simulate_feedbacks(system: MusicAppSystem, user_ids: list[int]) -> None:
    now = datetime.utcnow()
    with system._conn() as conn:
        for uid in user_ids:
            cnt = random.randint(0, 3)
            for _ in range(cnt):
                fb_time = (now - timedelta(days=random.randint(0, 60))).replace(microsecond=0)
                conn.execute(
                    """
                    INSERT INTO feedbacks (
                        user_id, feedback_time, rating, channel, feedback_type,
                        comment_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uid,
                        fb_time.isoformat(sep=" "),
                        random.randint(1, 5),
                        random.choice(["in_app", "email"]),
                        random.choice(["overall", "recommendation", "ui", "price"]),
                        random.choice(FEEDBACK_TEXTS),
                        _utcnow(),
                    ),
                )


def main():
    system = MusicAppSystem(str(DB_PATH))
    user_ids = simulate_users(system, n=800)
    simulate_preferences(system, user_ids)
    simulate_listening(system, user_ids)
    simulate_feedbacks(system, user_ids)
    for uid in user_ids:
        system.classify_user_segment(uid)
    print(f"Simulated data for {len(user_ids)} users into {DB_PATH}")


if __name__ == "__main__":
    main()
