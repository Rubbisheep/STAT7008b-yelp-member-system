from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "music.db"
OUTPUT_DIR = ROOT / "outputs"


def read_table(name: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {name}", conn)


def ensure_out():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def plot_register_trend(users: pd.DataFrame):
    users["day"] = pd.to_datetime(users["register_time"], errors="coerce").dt.date
    daily = users.groupby("day").size().reset_index(name="count")
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=daily, x="day", y="count", marker="o")
    plt.title("Daily New Users")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "register_trend.png")
    plt.close()


def plot_dau(listening: pd.DataFrame):
    listening["day"] = pd.to_datetime(listening["play_time"], errors="coerce").dt.date
    dau = listening.groupby("day")["user_id"].nunique().reset_index(name="dau")
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=dau, x="day", y="dau", marker="o", color="orange")
    plt.title("DAU (Listening)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dau.png")
    plt.close()


def plot_preferences(prefs: pd.DataFrame):
    def explode_column(series: pd.Series):
        tokens = []
        for val in series.dropna():
            tokens.extend([v.strip() for v in val.split(",") if v.strip()])
        return pd.Series(tokens)

    genres = explode_column(prefs["fav_genres"]).value_counts().reset_index()
    genres.columns = ["genre", "count"]
    plt.figure(figsize=(6, 4))
    sns.barplot(data=genres, x="genre", y="count")
    plt.title("Favorite Genres")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fav_genres.png")
    plt.close()


def plot_membership_vs_non(listening: pd.DataFrame, subs: pd.DataFrame):
    listening["date"] = pd.to_datetime(listening["play_time"], errors="coerce").dt.date
    active = listening.groupby(["user_id"]).agg(
        play_count=("log_id", "count"), duration=("duration_sec", "sum")
    ).reset_index()
    subs_active = subs[subs["status"] == "active"]
    active["is_member"] = active["user_id"].isin(subs_active["user_id"])
    grouped = active.groupby("is_member").agg(
        avg_play=("play_count", "mean"), avg_duration=("duration", "mean")
    ).reset_index()
    plt.figure(figsize=(6, 4))
    grouped_melt = grouped.melt(id_vars="is_member", value_vars=["avg_play", "avg_duration"])
    sns.barplot(data=grouped_melt, x="is_member", y="value", hue="variable")
    plt.title("Member vs Non-Member")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "member_vs_non.png")
    plt.close()


def plot_segments(seg_membership: pd.DataFrame, segments: pd.DataFrame):
    if seg_membership.empty:
        return
    merged = seg_membership.merge(segments, on="segment_id", how="left")
    counts = merged["segment_name"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    plt.figure(figsize=(6, 4))
    sns.barplot(data=counts, x="segment", y="count")
    plt.title("User Segments")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segments.png")
    plt.close()


def main():
    ensure_out()
    users = read_table("users")
    listening = read_table("listening_logs")
    prefs = read_table("user_preferences")
    subs = read_table("membership_subscriptions")
    seg_membership = read_table("user_segment_membership")
    segments = read_table("user_segments")

    plot_register_trend(users)
    plot_dau(listening)
    plot_preferences(prefs)
    plot_membership_vs_non(listening, subs)
    plot_segments(seg_membership, segments)
    print(f"Dashboard figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
