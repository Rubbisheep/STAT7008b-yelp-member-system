# scripts/advanced_analysis.py
"""
高级数据分析脚本（不修改数据库结构）

依赖：
    pip install pandas numpy matplotlib seaborn scikit-learn

运行方式（在项目根目录）：
    python -m scripts.advanced_analysis
"""

from pathlib import Path
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not installed; 聚类与流失模型部分将被跳过。")


# --------------------------------------------------------------------
# 全局配置
# --------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "music.db"
OUT_DIR = ROOT / "outputs" / "advanced"

plt.style.use("dark_background")
sns.set_theme(style="darkgrid", context="notebook")

plt.rcParams.update({
    "figure.facecolor": "#121212",
    "axes.facecolor": "#121212",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "0.9",
    "xtick.color": "0.8",
    "ytick.color": "0.8",
    "text.color": "0.95",
    "grid.color": "#444444",
    "grid.alpha": 0.3,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "legend.frameon": False,
})


def lowercase(listening_logs: pd.DataFrame) -> pd.DataFrame:
    """
    把 genre 统一转为小写
    """
    if listening_logs.empty or "genre" not in listening_logs.columns:
        return listening_logs

    normalized = listening_logs.copy()
    normalized["genre"] = listening_logs["genre"].map(
        lambda x: x.lower() if isinstance(x, str) else x
    )
    print(f"[INFO] Normalized genre to lowercase; {len(normalized)} rows kept.")
    return normalized


def beautify_axes(ax, title=None, xlabel=None, ylabel=None):
    """统一轴样式"""
    if title is not None:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return ax


def add_bar_labels(ax, fmt="{:.0f}", rotation=0, fontsize=9):
    """给条形图加数值标签"""
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            p.get_x() + p.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )
    return ax


def format_percent_axis(ax, axis="y", decimals=0):
    """把轴改成百分比形式"""
    from matplotlib.ticker import FuncFormatter

    if axis == "y":
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: f"{v * 100:.{decimals}f}%")
        )
    else:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: f"{v * 100:.{decimals}f}%")
        )
    return ax


def ensure_out(subdir: Optional[str] = None) -> Path:
    if subdir:
        d = OUT_DIR / subdir
    else:
        d = OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(filename: str, subdir: Optional[str] = None):
    out_dir = ensure_out(subdir)
    path = out_dir / filename
    plt.savefig(path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"[FIG] Saved {path}")


def read_table(name: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
            print(f"[INFO] Read table '{name}' with shape {df.shape}")
            return df
        except Exception as e:
            print(f"[WARN] Cannot read table '{name}': {e}")
            return pd.DataFrame()


def parse_dt(series: pd.Series, col_name: str) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", format="mixed")
    bad = dt.isna().sum()
    if bad > 0:
        print(f"[WARN] Column '{col_name}' has {bad} invalid timestamps; dropped.")
    return dt


# --------------------------------------------------------------------
# 一、构造分析用宽表：user_static, user_behavior
# --------------------------------------------------------------------

@dataclass
class Tables:
    users: pd.DataFrame
    user_preferences: pd.DataFrame
    listening_logs: pd.DataFrame
    feedbacks: pd.DataFrame
    feedback_topics: pd.DataFrame
    user_segments: pd.DataFrame
    user_segment_membership: pd.DataFrame
    user_profiles: pd.DataFrame


def load_all_tables() -> Tables:
    return Tables(
        users=read_table("users"),
        user_preferences=read_table("user_preferences"),
        listening_logs=read_table("listening_logs"),
        feedbacks=read_table("feedbacks"),
        feedback_topics=read_table("feedback_topics"),
        user_segments=read_table("user_segments"),
        user_segment_membership=read_table("user_segment_membership"),
        user_profiles=read_table("user_profiles"),
    )


def build_user_static(t: Tables) -> pd.DataFrame:
    """
    只做列合并，不改动 DB 结构。
    """
    users = t.users.copy()
    if users.empty:
        return pd.DataFrame()

    # 年龄（粗略按当前年份 2025 推算，可按需改）
    if "birth_year" in users.columns:
        users["age"] = 2025 - users["birth_year"].clip(1900, 2025)
    else:
        users["age"] = np.nan

    # 偏好表
    prefs = t.user_preferences.copy()
    if not prefs.empty:
        # 简单地保留 fav_genres / fav_scenes 文本
        prefs = prefs[["user_id", "fav_genres", "fav_scenes"]].drop_duplicates("user_id")
        users = users.merge(prefs, on="user_id", how="left")

    # 画像表
    profiles = t.user_profiles.copy()
    if not profiles.empty:
        keep_cols = [c for c in profiles.columns if c != "last_profile_update"]
        profiles = profiles[keep_cols]
        users = users.merge(profiles, on="user_id", how="left")

    # 分群
    seg_mem = t.user_segment_membership.copy()
    segments = t.user_segments.copy()
    if not seg_mem.empty and not segments.empty:
        seg = seg_mem.merge(segments, on="segment_id", how="left")
        seg = seg[["user_id", "segment_name"]].drop_duplicates("user_id")
        users = users.merge(seg, on="user_id", how="left")
    else:
        users["segment_name"] = np.nan

    print(f"[INFO] user_static shape: {users.shape}")
    return users


def build_user_behavior(
    listening_logs: pd.DataFrame,
    feedbacks: pd.DataFrame,
    window_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    基于 listening_logs 和 feedbacks 构建用户行为表，不写回 DB。

    window_days:
        None -> 全生命周期；
        int  -> 仅分析最近 window_days 天内的行为。
    """
    if listening_logs.empty:
        print("[WARN] listening_logs empty; user_behavior empty.")
        return pd.DataFrame()

    df = listening_logs.copy()
    dt = parse_dt(df["play_time"], "play_time")
    df = df.loc[~dt.isna(), :].copy()
    df["play_date"] = dt.dt.normalize()
    df["play_hour"] = dt.dt.hour
    df["weekday"] = dt.dt.day_name()
    df["is_night"] = df["play_hour"].between(20, 23)
    df["is_weekend"] = dt.dt.weekday >= 5  # 5,6

    if window_days is not None:
        max_day = df["play_date"].max()
        min_day = max_day - pd.Timedelta(days=window_days)
        df = df[df["play_date"].between(min_day, max_day)]
        print(f"[INFO] user_behavior 使用时间窗口: {min_day.date()} ~ {max_day.date()}")

    # 基础聚合
    agg = df.groupby("user_id").agg(
        total_plays=("user_id", "size"),
        active_days=("play_date", "nunique"),
        total_duration_sec=("duration_sec", "sum"),
        skip_count=("is_skipped", lambda s: (s == 1).sum()),
        recommend_plays=("from_recommend", lambda s: (s == 1).sum()),
        mobile_plays=("device_type", lambda s: (s == "mobile").sum()),
        genre_diversity=("genre", "nunique"),
        top_genre=("genre", lambda s: s.value_counts().idxmax() if len(s) > 0 else None),
        night_plays=("is_night", lambda s: (s == True).sum()),
        weekend_plays=("is_weekend", lambda s: (s == True).sum()),
    )

    # 比例 & 派生指标
    agg["duration_hours"] = agg["total_duration_sec"] / 3600.0
    agg["avg_session_sec"] = agg["total_duration_sec"] / agg["total_plays"].clip(lower=1)
    agg["skip_rate"] = agg["skip_count"] / agg["total_plays"].clip(lower=1)
    agg["recommend_ratio"] = agg["recommend_plays"] / agg["total_plays"].clip(lower=1)
    agg["mobile_ratio"] = agg["mobile_plays"] / agg["total_plays"].clip(lower=1)
    agg["night_ratio"] = agg["night_plays"] / agg["total_plays"].clip(lower=1)
    agg["weekend_ratio"] = agg["weekend_plays"] / agg["total_plays"].clip(lower=1)

    # 反馈聚合
    if not feedbacks.empty:
        fb = feedbacks.copy()
        fb_dt = parse_dt(fb["feedback_time"], "feedback_time")
        fb = fb.loc[~fb_dt.isna(), :]
        fb["neg_flag"] = fb["rating"] <= 2

        fb_agg = fb.groupby("user_id").agg(
            feedback_cnt=("user_id", "size"),
            avg_rating=("rating", "mean"),
            neg_feedback_cnt=("neg_flag", "sum"),
            first_feedback_time=("feedback_time", "min"),
            last_feedback_time=("feedback_time", "max"),
        )

        agg = agg.merge(fb_agg, on="user_id", how="left")
    else:
        agg["feedback_cnt"] = 0
        agg["avg_rating"] = np.nan
        agg["neg_feedback_cnt"] = 0

    agg.reset_index(inplace=True)
    print(f"[INFO] user_behavior shape: {agg.shape}")
    return agg


# --------------------------------------------------------------------
# 二、用户结构画像分析
# --------------------------------------------------------------------

def analyze_user_structure(user_static: pd.DataFrame):
    if user_static.empty:
        print("[INFO] user_static empty; skip user structure analysis.")
        return

    out_sub = "user_structure"

    # 性别分布
    if "gender" in user_static.columns:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        vc = user_static["gender"].value_counts(dropna=False)
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette="crest")
        beautify_axes(ax, "Gender distribution", "Gender", "Users")
        add_bar_labels(ax)
        save_fig("gender_distribution.png", out_sub)

    # 年龄段
    if "age" in user_static.columns:
        bins = [0, 18, 24, 30, 40, 60, 120]
        labels = ["<18", "18-24", "25-30", "31-40", "41-60", "60+"]
        age_cut = pd.cut(user_static["age"], bins=bins, labels=labels, right=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        vc = age_cut.value_counts().sort_index()
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette="rocket")
        beautify_axes(ax, "Age bands", "Age band", "Users")
        add_bar_labels(ax)
        save_fig("age_bands.png", out_sub)

    # 地区 top10
    if "region" in user_static.columns:
        top_region = user_static["region"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            x=top_region.values[::-1],
            y=top_region.index[::-1].astype(str),
            ax=ax,
            palette="mako",
        )
        beautify_axes(ax, "Top 10 regions", "Users", "Region")
        add_bar_labels(ax)
        save_fig("region_top10.png", out_sub)

    # 注册渠道
    if "register_source" in user_static.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        vc = user_static["register_source"].value_counts(dropna=False)
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette="flare")
        beautify_axes(ax, "Register source distribution", "Source", "Users")
        ax.tick_params(axis="x", rotation=30)
        add_bar_labels(ax)
        save_fig("register_source.png", out_sub)

    # 渠道 × 性别 交叉表（如有）
    if "gender" in user_static.columns and "register_source" in user_static.columns:
        ct = pd.crosstab(
            user_static["register_source"], user_static["gender"], normalize="index"
        )

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.heatmap(
            ct,
            annot=True,
            fmt=".0%",
            cmap="YlGnBu",
            cbar_kws={"label": "Row percentage"},
            linewidths=0.3,
            linecolor="#333333",
            ax=ax,
        )
        ax.set_xlabel("Gender")
        ax.set_ylabel("Register source")
        ax.set_title("Register source × Gender (row %)", fontsize=13, fontweight="bold")
        save_fig("source_by_gender.png", out_sub)

def plot_user_scenario(
    user_id: int,
    listening_logs: pd.DataFrame,
    subdir: str = "user_scenarios",
):
    """
    为单个用户画 2 张图：
      1. 24 小时内各个小时的播放次数分布
      2. Weekday × Hour 的播放热力图
    """
    df = listening_logs[listening_logs["user_id"] == user_id].copy()
    if df.empty:
        print(f"[INFO] No logs for user {user_id}; skip scenario plot.")
        return

    dt = parse_dt(df["play_time"], "play_time")
    df = df.loc[~dt.isna(), :].copy()
    if df.empty:
        print(f"[INFO] No valid timestamps for user {user_id}.")
        return

    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.day_name()

    # 图1：24 小时听歌分布
    hour_counts = (
        df.groupby("hour")["user_id"]
        .size()
        .reindex(range(24), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        x=hour_counts.index,
        y=hour_counts.values,
        ax=ax,
        color=sns.color_palette("coolwarm")[2],
    )
    beautify_axes(
        ax,
        f"Hourly listening pattern (user {user_id})",
        "Hour of day",
        "Play count",
    )
    add_bar_labels(ax)
    save_fig(f"user_{user_id}_hourly_pattern.png", subdir)

    # 图2：Weekday × Hour 热力图
    # 把 weekday 按顺序排一下
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)

    pivot = (
        df.groupby(["weekday", "hour"])["user_id"]
        .size()
        .reset_index(name="plays")
        .pivot(index="weekday", columns="hour", values="plays")
        .reindex(index=weekday_order)
        .reindex(columns=range(24), fill_value=0)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        pivot,
        cmap="magma",
        linewidths=0.2,
        linecolor="#333333",
        cbar_kws={"label": "Play count"},
        ax=ax,
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Weekday")
    ax.set_title(
        f"Weekday × hour heatmap (user {user_id})",
        fontsize=13,
        fontweight="bold",
    )
    save_fig(f"user_{user_id}_weekday_hour_heatmap.png", subdir)



# --------------------------------------------------------------------
# 三、行为强度与使用习惯分析
# --------------------------------------------------------------------
        
def add_activity_level(user_behavior: pd.DataFrame) -> pd.DataFrame:
    """根据 duration_hours 把用户分成 light / medium / heavy 三档"""
    if user_behavior.empty or "duration_hours" not in user_behavior.columns:
        user_behavior["activity_level"] = "unknown"
        return user_behavior

    duration = user_behavior["duration_hours"].fillna(0)
    q1, q3 = duration.quantile([0.33, 0.66])

    def tag(x):
        if x < q1:
            return "light"
        elif x < q3:
            return "medium"
        else:
            return "heavy"

    user_behavior["activity_level"] = duration.apply(tag)
    return user_behavior


def analyze_behavior_patterns(user_behavior: pd.DataFrame, user_static: pd.DataFrame):
    if user_behavior.empty:
        print("[INFO] user_behavior empty; skip behavior analysis.")
        return

    out_sub = "behavior_patterns"
    user_behavior = add_activity_level(user_behavior)

    # 播放总时长分布（对数 y 轴）
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(
        user_behavior["duration_hours"],
        bins=50,
        log_scale=(False, True),
        edgecolor="#222222",
        linewidth=0.3,
        ax=ax,
        color="#4DB6AC",
    )
    beautify_axes(ax, "Distribution of total listening hours", "Hours", "Users (log-y)")
    save_fig("duration_hours_hist.png", out_sub)

    # 活跃天数分布
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(
        user_behavior["active_days"],
        bins=30,
        edgecolor="#222222",
        linewidth=0.3,
        ax=ax,
        color="#9575CD",
    )
    beautify_axes(ax, "Distribution of active days", "Active days", "Users")
    save_fig("active_days_hist.png", out_sub)

    # 活跃层级占比
    fig, ax = plt.subplots(figsize=(5.5, 4))
    vc = user_behavior["activity_level"].value_counts()
    sns.barplot(
        x=vc.index.astype(str),
        y=vc.values,
        ax=ax,
        palette=["#66BB6A", "#FFCA28", "#EF5350"],
    )
    beautify_axes(ax, "Activity levels", "Level", "Users")
    add_bar_labels(ax)
    save_fig("activity_level_distribution.png", out_sub)

    # 各层级平均行为 CSV：不改
    cols = ["duration_hours", "active_days", "genre_diversity", "skip_rate",
            "recommend_ratio", "mobile_ratio", "night_ratio", "weekend_ratio"]
    summary = user_behavior.groupby("activity_level")[cols].mean()
    summary.to_csv(ensure_out(out_sub) / "activity_level_summary.csv")
    print("[INFO] Saved activity_level_summary.csv")

    # skip_rate 分布
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(
        user_behavior["skip_rate"],
        bins=40,
        edgecolor="#222222",
        linewidth=0.3,
        ax=ax,
        color="#F06292",
    )
    beautify_axes(ax, "Distribution of skip rate", "Skip rate", "Users")
    save_fig("skip_rate_hist.png", out_sub)

    # 设备偏好分布
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(
        user_behavior["mobile_ratio"],
        bins=40,
        edgecolor="#222222",
        linewidth=0.3,
        ax=ax,
        color="#4FC3F7",
    )
    beautify_axes(ax, "Distribution of mobile usage", "Mobile plays ratio", "Users")
    save_fig("mobile_ratio_hist.png", out_sub)

    # 不同 activity_level 的夜间占比
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(
        data=user_behavior,
        x="activity_level",
        y="night_ratio",
        palette=["#66BB6A", "#FFCA28", "#EF5350"],
        ax=ax,
    )
    beautify_axes(ax, "Night usage by activity level", "Activity level", "Night ratio")
    format_percent_axis(ax, axis="y", decimals=0)
    save_fig("night_ratio_by_level.png", out_sub)



# --------------------------------------------------------------------
# 四、偏好与内容分析
# --------------------------------------------------------------------
    
def analyze_preferences(
    listening_logs: pd.DataFrame,
    user_behavior: pd.DataFrame,
    user_static: pd.DataFrame
):
    if listening_logs.empty:
        print("[INFO] listening_logs empty; skip preference analysis.")
        return

    out_sub = "preferences"

    df = listening_logs.copy()
    dt = parse_dt(df["play_time"], "play_time")
    df = df.loc[~dt.isna(), :]

    # genre 结构
    if "genre" in df.columns:
        genre_counts = df["genre"].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            x=genre_counts.values[::-1],
            y=genre_counts.index[::-1].astype(str),
            ax=ax,
            palette="viridis",
        )
        beautify_axes(ax, "Top 20 genres by play count", "Plays", "Genre")
        add_bar_labels(ax)
        save_fig("genre_top20.png", out_sub)

    # genre_diversity 分布
    if "genre_diversity" in user_behavior.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.histplot(
            user_behavior["genre_diversity"],
            bins=30,
            edgecolor="#222222",
            linewidth=0.3,
            ax=ax,
            color="#FFB74D",
        )
        beautify_axes(ax, "Genre diversity distribution",
                      "Distinct genres per user", "Users")
        save_fig("genre_diversity_hist.png", out_sub)

    # top_genre × activity_level 交叉
    if "top_genre" in user_behavior.columns and "activity_level" in user_behavior.columns:
        cross = pd.crosstab(
            user_behavior["top_genre"],
            user_behavior["activity_level"],
            normalize="columns",
        )
        cross = cross.sort_values("light", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(7.5, 5))
        sns.heatmap(
            cross,
            annot=True,
            fmt=".0%",
            cmap="rocket_r",
            linewidths=0.3,
            linecolor="#333333",
            cbar_kws={"label": "Column percentage"},
            ax=ax,
        )
        ax.set_xlabel("Activity level")
        ax.set_ylabel("Top genre")
        ax.set_title("Top genre × activity level (col %)",
                     fontsize=13, fontweight="bold")
        save_fig("top_genre_by_activity_level.png", out_sub)



# --------------------------------------------------------------------
# 五、留存与生命周期分析（简单版）
# --------------------------------------------------------------------

def compute_daily_retention(
    users: pd.DataFrame,
    listening_logs: pd.DataFrame,
    horizons: List[int] = [1, 7, 30],
) -> pd.DataFrame:
    if users.empty or listening_logs.empty:
        return pd.DataFrame()

    reg_dt = parse_dt(users["register_time"], "register_time")
    u = users.loc[~reg_dt.isna(), ["user_id"]].copy()
    u["reg_day"] = reg_dt[~reg_dt.isna()].dt.normalize()

    play_dt = parse_dt(listening_logs["play_time"], "play_time")
    l = listening_logs.loc[~play_dt.isna(), ["user_id"]].copy()
    l["play_day"] = play_dt[~play_dt.isna()].dt.normalize()

    # 注册当天的用户集合
    cohorts = u.groupby("reg_day")["user_id"].apply(set)

    # 每天的活跃用户集合
    active = l.groupby("play_day")["user_id"].apply(set)

    results = []
    for reg_day, users_set in cohorts.items():
        size = len(users_set)
        if size == 0:
            continue
        for h in horizons:
            target_day = reg_day + pd.Timedelta(days=h)
            act_set = active.get(target_day, set())
            retained = len(users_set & act_set)
            results.append({
                "reg_day": reg_day,
                "horizon": h,
                "cohort_size": size,
                "retained": retained,
                "retention_rate": retained / size
            })

    return pd.DataFrame(results)


def analyze_retention(users: pd.DataFrame, listening_logs: pd.DataFrame):
    if users.empty or listening_logs.empty:
        print("[INFO] users/listening empty; skip retention.")
        return

    out_sub = "retention"
    df = compute_daily_retention(users, listening_logs)
    if df.empty:
        print("[INFO] retention df empty.")
        return

    summary = (
        df.groupby("horizon")["retention_rate"]
        .mean()
        .reset_index()
        .sort_values("horizon")
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=summary, x="horizon", y="retention_rate",
                ax=ax, palette="Blues")
    beautify_axes(ax, "Average D1 / D7 / D30 retention",
                  "Days since registration", "Retention rate")
    format_percent_axis(ax, axis="y", decimals=0)
    add_bar_labels(ax, fmt="{:.0%}")
    save_fig("avg_retention_d1_d7_d30.png", out_sub)

    summary.to_csv(ensure_out(out_sub) / "avg_retention_d1_d7_d30.csv", index=False)
    print("[INFO] Saved avg_retention_d1_d7_d30.csv")



# --------------------------------------------------------------------
# 六、反馈与文本分析（基于现有 feedbacks / feedback_topics）
# --------------------------------------------------------------------

def analyze_feedback(
    feedbacks: pd.DataFrame,
    feedback_topics: pd.DataFrame,
    user_behavior: pd.DataFrame,
    user_static: pd.DataFrame
):
    if feedbacks.empty:
        print("[INFO] feedbacks empty; skip feedback analysis.")
        return

    out_sub = "feedback"

    # rating 分布
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.histplot(
        feedbacks["rating"],
        bins=5,
        discrete=True,
        edgecolor="#222222",
        linewidth=0.4,
        ax=ax,
        color="#FFCA28",
    )
    beautify_axes(ax, "Rating distribution", "Rating", "Count")
    save_fig("rating_hist.png", out_sub)

    # 渠道差异
    if "channel" in feedbacks.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.boxplot(
            data=feedbacks,
            x="channel",
            y="rating",
            palette="Set2",
            ax=ax,
        )
        beautify_axes(ax, "Rating by channel", "Channel", "Rating")
        ax.tick_params(axis="x", rotation=25)
        save_fig("rating_by_channel.png", out_sub)

    # topic × rating
    if not feedback_topics.empty:
        ft = feedback_topics.copy()
        merged = ft.merge(
            feedbacks[["feedback_id", "rating"]],
            on="feedback_id",
            how="left",
        )
        topic_summary = (
            merged.groupby("topic_label")
            .agg(
                avg_rating=("rating", "mean"),
                cnt=("feedback_id", "size"),
            )
            .sort_values("cnt", ascending=False)
        )
        topic_summary.to_csv(
            ensure_out(out_sub) / "topic_rating_summary.csv", index=False
        )
        print("[INFO] Saved topic_rating_summary.csv")

        top_topics = topic_summary.head(10).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=top_topics,
            x="avg_rating",
            y="topic_label",
            ax=ax,
            palette="coolwarm",
        )
        beautify_axes(ax, "Top topics by count (avg rating)",
                      "Average rating", "Topic")
        add_bar_labels(ax, fmt="{:.2f}")
        save_fig("topic_avg_rating_top10.png", out_sub)

    # 用户层面情感画像
    fb_user = feedbacks.groupby("user_id").agg(
        user_avg_rating=("rating", "mean"),
        feedback_cnt=("user_id", "size"),
        low_rating_cnt=("rating", lambda s: (s <= 2).sum()),
    )
    fb_user.reset_index(inplace=True)

    if not user_behavior.empty:
        ub = user_behavior.merge(fb_user, on="user_id", how="left")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.scatterplot(
            data=ub,
            x="skip_rate",
            y="user_avg_rating",
            alpha=0.35,
            s=18,
            edgecolor="none",
            color="#4FC3F7",
            ax=ax,
        )
        beautify_axes(ax, "Skip rate vs user avg rating",
                      "Skip rate", "User average rating")
        save_fig("skip_rate_vs_user_avg_rating.png", out_sub)



# --------------------------------------------------------------------
# 七、行为聚类（不写回 DB，只在分析代码里用）
# --------------------------------------------------------------------

def perform_clustering(user_behavior: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not SKLEARN_AVAILABLE:
        print("[INFO] scikit-learn not available; skip clustering.")
        return None

    if user_behavior.empty:
        print("[INFO] user_behavior empty; skip clustering.")
        return None

    out_sub = "clustering"

    cols = [
        "active_days",
        "duration_hours",
        "genre_diversity",
        "skip_rate",
        "recommend_ratio",
        "night_ratio",
        "weekend_ratio",
    ]
    df = user_behavior.copy()
    X = df[cols].fillna(0.0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=cols)
    centers_df["cluster"] = range(4)
    centers_df.to_csv(ensure_out(out_sub) / "cluster_centers.csv", index=False)
    print("[INFO] Saved cluster_centers.csv")

    # 可视化两个维度
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=df,
        x="duration_hours",
        y="genre_diversity",
        hue="cluster",
        palette="tab10",
        alpha=0.45,
        s=20,
        ax=ax,
    )
    beautify_axes(
        ax,
        "Clusters on duration vs genre diversity",
        "Total listening hours",
        "Distinct genres",
    )
    ax.legend(title="Cluster", fontsize=9)
    save_fig("clusters_duration_vs_genre.png", out_sub)

    return df



# --------------------------------------------------------------------
# 八、简单流失分析与基线模型（仅在代码中构造标签）
# --------------------------------------------------------------------

def label_churn(users: pd.DataFrame, listening_logs: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    定义：观测窗口最后 window_days 内无播放行为 -> 视为 churn=1
    返回含 user_id, is_churn 的 DataFrame。
    """
    if users.empty or listening_logs.empty:
        return pd.DataFrame(columns=["user_id", "is_churn"])

    play_dt = parse_dt(listening_logs["play_time"], "play_time")
    l = listening_logs.loc[~play_dt.isna(), ["user_id"]].copy()
    l["play_day"] = play_dt[~play_dt.isna()].dt.normalize()

    last_day = l["play_day"].max()
    cutoff = last_day - pd.Timedelta(days=window_days)

    last_play = l.groupby("user_id")["play_day"].max().rename("last_play_day")
    churn_df = users[["user_id"]].merge(last_play, on="user_id", how="left")

    churn_df["is_churn"] = churn_df["last_play_day"].apply(
        lambda d: 1 if (pd.isna(d) or d < cutoff) else 0
    )
    return churn_df[["user_id", "is_churn"]]


def analyze_churn_model(
    users: pd.DataFrame,
    listening_logs: pd.DataFrame,
    user_behavior: pd.DataFrame,
    user_static: pd.DataFrame
):
    if not SKLEARN_AVAILABLE:
        print("[INFO] scikit-learn not available; skip churn model.")
        return

    if users.empty or listening_logs.empty or user_behavior.empty:
        print("[INFO] Missing data for churn model.")
        return

    out_sub = "churn_model"

    churn_df = label_churn(users, listening_logs, window_days=30)
    df = user_behavior.merge(churn_df, on="user_id", how="inner")
    # 简单过滤掉极少播放用户，避免噪音
    df = df[df["total_plays"] >= 3]

    if df["is_churn"].nunique() < 2:
        print("[INFO] Only one class in churn label; skip model.")
        return

    features = [
        "active_days",
        "duration_hours",
        "genre_diversity",
        "skip_rate",
        "recommend_ratio",
        "mobile_ratio",
        "night_ratio",
        "weekend_ratio",
        "feedback_cnt",
        "neg_feedback_cnt",
    ]
    X = df[features].fillna(0.0).values
    y = df["is_churn"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 简单地用全部样本做训练+评估（可以扩展为 train/test split）
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"[RESULT] Churn model AUC (insample) = {auc:.3f}")

    coef_df = pd.DataFrame({
        "feature": features,
        "coef": model.coef_[0]
    }).sort_values("coef", key=lambda s: s.abs(), ascending=False)
    coef_df.to_csv(ensure_out(out_sub) / "churn_logit_coef.csv", index=False)
    print("[INFO] Saved churn_logit_coef.csv")


# --------------------------------------------------------------------
# 九、题材相似度分析（基于用户级共现）
# --------------------------------------------------------------------

def analyze_genre_similarity(listening_logs: pd.DataFrame, top_n: int = 20):
    """
    基于“用户级共现”构造 genre 共现和相似度矩阵，并画热力图。
    不修改表结构，只用 listening_logs 中的 user_id, genre。
    """
    if listening_logs.empty or "genre" not in listening_logs.columns:
        print("[INFO] listening_logs empty or no 'genre'; skip genre similarity.")
        return

    out_sub = "genre_similarity"

    df = listening_logs[["user_id", "genre"]].dropna().copy()
    df["genre"] = df["genre"].astype(str)

    # 选取最常见的前 top_n 个题材，保证图像可读
    top_genres = df["genre"].value_counts().head(top_n).index.tolist()
    df_top = df[df["genre"].isin(top_genres)]

    if df_top.empty:
        print("[INFO] No logs after filtering top genres; skip.")
        return

    # 用户级别的 genre 集合
    user_genres = (
        df_top.groupby("user_id")["genre"]
        .apply(lambda s: sorted(set(s)))
        .reset_index()
    )

    # 初始化共现矩阵
    genres = top_genres
    idx = {g: i for i, g in enumerate(genres)}
    n = len(genres)
    co_mat = np.zeros((n, n), dtype=int)

    # 统计共现：每个用户的 genre 集合中，所有 (i,j) 对各加 1
    for row in user_genres["genre"]:
        if len(row) <= 1:
            continue
        for i in range(len(row)):
            gi = row[i]
            ii = idx[gi]
            # 包含自身对角：ii,ii 表示有这个 genre 的用户数
            co_mat[ii, ii] += 1
            for j in range(i + 1, len(row)):
                gj = row[j]
                jj = idx[gj]
                co_mat[ii, jj] += 1
                co_mat[jj, ii] += 1

    co_df = pd.DataFrame(co_mat, index=genres, columns=genres)
    co_df.to_csv(ensure_out(out_sub) / "genre_cooccurrence_matrix.csv")
    print("[INFO] Saved genre_cooccurrence_matrix.csv")

    # 基于 Jaccard 思路计算相似度：co_ij / (co_ii + co_jj - co_ij)
    sim_mat = np.zeros_like(co_mat, dtype=float)
    diag = np.diag(co_mat)

    for i in range(n):
        for j in range(n):
            co_ij = co_mat[i, j]
            denom = diag[i] + diag[j] - co_ij
            sim_mat[i, j] = co_ij / denom if denom > 0 else 0.0

    sim_df = pd.DataFrame(sim_mat, index=genres, columns=genres)
    sim_df.to_csv(ensure_out(out_sub) / "genre_similarity_matrix.csv")
    print("[INFO] Saved genre_similarity_matrix.csv")

    # 画相似度热力图
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        sim_df,
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Jaccard similarity"},
        ax=ax,
    )
    ax.set_xlabel("Genre")
    ax.set_ylabel("Genre")
    ax.set_title("Genre similarity (user-level Jaccard)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_fig("genre_similarity_heatmap.png", out_sub)


# --------------------------------------------------------------------
# 十、分群偏好与贡献分析（基于 segment_name）
# --------------------------------------------------------------------

def analyze_segment_preferences(
    user_static: pd.DataFrame,
    user_behavior: pd.DataFrame,
    listening_logs: pd.DataFrame,
):
    """
    基于 user_static 中的 segment_name，对不同 segment 的价值贡献和内容偏好做可视化。
    """
    if user_static.empty or user_behavior.empty:
        print("[INFO] user_static or user_behavior empty; skip segment analysis.")
        return

    if "segment_name" not in user_static.columns:
        print("[INFO] 'segment_name' not in user_static; skip segment analysis.")
        return

    out_sub = "segment_preferences"

    # 合并行为 + 分群信息
    ub = user_behavior.merge(
        user_static[["user_id", "segment_name"]],
        on="user_id",
        how="left",
    )
    ub = ub[~ub["segment_name"].isna()].copy()

    if ub.empty:
        print("[INFO] No users with segment_name; skip.")
        return

    # 1) 各 segment 的价值与行为指标
    seg_metrics = (
        ub.groupby("segment_name")
        .agg(
            user_count=("user_id", "nunique"),
            total_duration_hours=("duration_hours", "sum"),
            avg_duration_hours=("duration_hours", "mean"),
            avg_active_days=("active_days", "mean"),
            avg_skip_rate=("skip_rate", "mean"),
            avg_rating=("avg_rating", "mean"),
        )
        .sort_values("total_duration_hours", ascending=False)
    )
    seg_metrics.to_csv(
        ensure_out(out_sub) / "segment_behavior_metrics.csv"
    )
    print("[INFO] Saved segment_behavior_metrics.csv")

    # 图1：各 segment 总听歌时长（贡献）
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        x=seg_metrics.index.astype(str),
        y=seg_metrics["total_duration_hours"].values,
        ax=ax,
        color=sns.color_palette("Blues")[3],
    )
    beautify_axes(
        ax,
        "Total listening hours by segment",
        "Segment",
        "Total listening hours",
    )
    add_bar_labels(ax, fmt="{:.1f}", rotation=0)
    ax.tick_params(axis="x", rotation=20)
    save_fig("segment_total_duration.png", out_sub)

    # 图2：各 segment 平均用户听歌时长
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        x=seg_metrics.index.astype(str),
        y=seg_metrics["avg_duration_hours"].values,
        ax=ax,
        color=sns.color_palette("Greens")[3],
    )
    beautify_axes(
        ax,
        "Average listening hours per user by segment",
        "Segment",
        "Avg hours per user",
    )
    add_bar_labels(ax, fmt="{:.2f}", rotation=0)
    ax.tick_params(axis="x", rotation=20)
    save_fig("segment_avg_duration.png", out_sub)

    # 2) segment × genre 偏好热力图（使用 listening_logs）
    if listening_logs.empty or "genre" not in listening_logs.columns:
        print("[INFO] listening_logs empty or no genre; skip segment×genre heatmap.")
        return

    logs = listening_logs.merge(
        user_static[["user_id", "segment_name"]],
        on="user_id",
        how="left",
    )
    logs = logs[~logs["segment_name"].isna()].copy()
    logs["genre"] = logs["genre"].astype(str)

    # 只取播放数 topK 的 genre，保证图不至于太乱
    top_genres = (
        logs["genre"].value_counts()
        .head(10)
        .index.tolist()
    )
    logs_top = logs[logs["genre"].isin(top_genres)]

    if logs_top.empty:
        print("[INFO] No logs for top genres; skip.")
        return

    seg_genre_counts = (
        logs_top.groupby(["segment_name", "genre"])["user_id"]
        .size()
        .rename("plays")
        .reset_index()
    )

    # 行归一：每个 segment 内不同 genre 占比
    total_per_seg = (
        seg_genre_counts.groupby("segment_name")["plays"].transform("sum")
    )
    seg_genre_counts["share"] = seg_genre_counts["plays"] / total_per_seg

    heat = seg_genre_counts.pivot(
        index="segment_name", columns="genre", values="share"
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".0%",
        cmap="rocket_r",
        cbar_kws={"label": "Share of plays (row %)"},
        linewidths=0.3,
        linecolor="#333333",
        ax=ax,
    )
    ax.set_xlabel("Genre")
    ax.set_ylabel("Segment")
    ax.set_title(
        "Genre preference by segment (row %)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    save_fig("segment_genre_preference_heatmap.png", out_sub)


# --------------------------------------------------------------------
# 十一、典型用户场景画像（1–2 个案例）
# --------------------------------------------------------------------

def analyze_user_scenarios(
    user_behavior: pd.DataFrame,
    listening_logs: pd.DataFrame,
    user_static: pd.DataFrame,
    num_users: int = 2,
):
    """
    自动选取若干典型用户（默认 2 个：一个重度、一个轻度），画场景画像。
    """
    if user_behavior.empty or listening_logs.empty:
        print("[INFO] user_behavior or listening_logs empty; skip user scenarios.")
        return

    # 过滤掉 duration_hours = 0 的用户
    ub = user_behavior[user_behavior["duration_hours"] > 0].copy()
    if ub.empty:
        print("[INFO] No users with positive duration; skip user scenarios.")
        return

    # 按总时长排序
    ub_sorted = ub.sort_values("duration_hours")

    # 选一个轻度用户（duration_hours 最小）
    light_user = int(ub_sorted.iloc[0]["user_id"])

    # 选一个重度用户（duration_hours 最大）
    heavy_user = int(ub_sorted.iloc[-1]["user_id"])

    user_ids = [light_user, heavy_user]
    user_ids = list(dict.fromkeys(user_ids))  # 去重

    print(f"[INFO] User scenarios will be generated for users: {user_ids}")

    for uid in user_ids:
        plot_user_scenario(uid, listening_logs, subdir="user_scenarios")




# --------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------

def main():
    print(f"[INFO] DB_PATH = {DB_PATH}")
    ensure_out()

    # 1. 读取表
    t = load_all_tables()
    
    # 2. 把 genre 统一转为小写
    logs_lowercase = lowercase(t.listening_logs)

    # 3. 构造宽表（基于去掉 pop 的 listening_logs）
    user_static = build_user_static(t)
    user_behavior = build_user_behavior(logs_lowercase, t.feedbacks, window_days=None)
    # 3. 原有分析模块
    analyze_user_structure(user_static)
    analyze_behavior_patterns(user_behavior, user_static)
    analyze_preferences(t.listening_logs, user_behavior, user_static)
    analyze_retention(t.users, t.listening_logs)
    analyze_feedback(t.feedbacks, t.feedback_topics, user_behavior, user_static)

    clustered = perform_clustering(user_behavior)
    if clustered is not None:
        # 如有需要，可以把 cluster 信息与 user_static 合并做进一步分析
        pass

    analyze_churn_model(t.users, t.listening_logs, user_behavior, user_static)

    # 4. 新增：题材相似度、分群偏好与贡献、典型用户场景画像
    analyze_genre_similarity(t.listening_logs, top_n=20)
    analyze_segment_preferences(user_static, user_behavior, t.listening_logs)
    analyze_user_scenarios(user_behavior, t.listening_logs, user_static, num_users=2)

    print(f"[INFO] Advanced analysis finished. See outputs in {OUT_DIR}")



if __name__ == "__main__":
    main()
