PRAGMA foreign_keys = ON;

-- Core user and membership tables
CREATE TABLE IF NOT EXISTS users (
  user_id            INTEGER PRIMARY KEY,
  email              TEXT,
  phone              TEXT,
  nickname           TEXT,
  gender             TEXT,                -- 'M'/'F'/'O'/NULL
  birth_year         INTEGER,
  region             TEXT,
  register_source    TEXT,
  register_time      TEXT NOT NULL,
  invited_by_user_id INTEGER,
  status             TEXT DEFAULT 'active',
  created_at         TEXT,
  updated_at         TEXT
);

CREATE TABLE IF NOT EXISTS membership_plans (
  plan_id       INTEGER PRIMARY KEY,
  plan_name     TEXT,
  price_monthly REAL,
  description   TEXT
);

CREATE TABLE IF NOT EXISTS membership_subscriptions (
  sub_id        INTEGER PRIMARY KEY,
  user_id       INTEGER NOT NULL,
  plan_id       INTEGER,
  status        TEXT,
  start_at      TEXT,
  end_at        TEXT,
  auto_renew    INTEGER,
  cancel_reason TEXT,
  created_at    TEXT,
  updated_at    TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id),
  FOREIGN KEY (plan_id) REFERENCES membership_plans(plan_id)
);

CREATE TABLE IF NOT EXISTS user_preferences (
  user_id        INTEGER PRIMARY KEY,
  fav_genres     TEXT,   -- comma separated: 'pop,rock,jazz'
  fav_scenes     TEXT,   -- 'study,commute,workout'
  dislike_genres TEXT,
  listening_goal TEXT,
  extra_info     TEXT,
  created_at     TEXT,
  updated_at     TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Behaviors and feedback
CREATE TABLE IF NOT EXISTS listening_logs (
  log_id        INTEGER PRIMARY KEY,
  user_id       INTEGER NOT NULL,
  play_time     TEXT NOT NULL,
  duration_sec  INTEGER,
  track_id      INTEGER,
  genre         TEXT,
  is_skipped    INTEGER,
  device_type   TEXT,
  from_recommend INTEGER,
  created_at    TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS feedbacks (
  feedback_id   INTEGER PRIMARY KEY,
  user_id       INTEGER NOT NULL,
  feedback_time TEXT NOT NULL,
  rating        INTEGER,
  channel       TEXT,
  feedback_type TEXT,
  comment_text  TEXT,
  created_at    TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Profiles and segmentation
CREATE TABLE IF NOT EXISTS user_profiles (
  user_id             INTEGER PRIMARY KEY,
  mbti_guess          TEXT,
  main_interest       TEXT,
  active_time_pattern TEXT,
  spending_level      TEXT,
  churn_risk_level    TEXT,
  keywords_summary    TEXT,
  last_profile_update TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS user_segments (
  segment_id   INTEGER PRIMARY KEY,
  segment_name TEXT,
  description  TEXT
);

CREATE TABLE IF NOT EXISTS user_segment_membership (
  user_id     INTEGER NOT NULL,
  segment_id  INTEGER NOT NULL,
  assigned_by TEXT,
  assigned_at TEXT,
  PRIMARY KEY (user_id, segment_id),
  FOREIGN KEY (user_id) REFERENCES users(user_id),
  FOREIGN KEY (segment_id) REFERENCES user_segments(segment_id)
);

CREATE TABLE IF NOT EXISTS feedback_topics (
  feedback_id   INTEGER NOT NULL,
  topic_label   TEXT NOT NULL,
  sentiment_score REAL,
  keywords        TEXT,
  PRIMARY KEY (feedback_id, topic_label),
  FOREIGN KEY (feedback_id) REFERENCES feedbacks(feedback_id)
);
