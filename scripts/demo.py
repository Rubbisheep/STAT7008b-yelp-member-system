from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.simulate_data import main as simulate_main  # noqa: E402
from src.music_app_system import MusicAppSystem  # noqa: E402


DB_PATH = ROOT / "data" / "music.db"


def demo_flow():
    # 1) Ensure data exists
    simulate_main()

    system = MusicAppSystem(str(DB_PATH))

    # 2) Register a new user
    new_uid = system.register_user(
        email="demo@example.com",
        phone=None,
        nickname="DemoUser",
        gender="F",
        birth_year=2000,
        region="DemoCity",
        register_source="demo_flow",
    )

    # 3) Generate a few activity and feedback records
    system.log_listening_session(
        user_id=new_uid,
        play_time=datetime.utcnow().isoformat(sep=" "),
        duration_sec=180,
        genre="pop",
        is_skipped=False,
        device_type="mobile",
    )
    system.log_feedback(
        user_id=new_uid,
        feedback_time=datetime.utcnow().isoformat(sep=" "),
        rating=5,
        comment_text="UI feels great and playlists work well",
    )

    # 4) Text mining + behavior profiling
    system.update_user_profile_from_text(new_uid)
    system.update_user_profile_from_behavior(new_uid)
    system.classify_user_segment(new_uid)

    profile = system.get_user_info(new_uid)
    print(f"Demo user created: {profile}")
    print("Profile enrichment done. Check user_profiles and user_segment_membership tables.")


if __name__ == "__main__":
    demo_flow()
