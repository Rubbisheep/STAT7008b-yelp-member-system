# Music Membership Analytics Project

[English](#english) | [中文](README_zh.md)

<p>
  <a href="README_zh.md" style="display:inline-block;padding:8px 12px;border:1px solid #ccc;border-radius:6px;text-decoration:none;">中文版本</a>
</p>

End-to-end pipeline: schema -> simulate data -> class library -> text mining stubs -> dashboard -> static pages.

---

## English

### Prerequisites
- Python 3.9+ (Anaconda env `workingenvs` is fine)
- No external DB (SQLite only)

### Quickstart (from repo root)
```bash
# 0) create/activate venv and install deps
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt

# 1) init DB
python scripts/init_db.py

# 2) simulate data (users, prefs, listening logs, feedbacks, segments)
python -m scripts.simulate_data

# 3) run demo (register user -> logs -> text mining -> segmentation)
python -m scripts.demo

# 4) generate dashboard charts (PNG to outputs/)
python scripts/generate_dashboard.py

# 5) view static pages (optional)
python -m http.server 8000
# open http://localhost:8000/static/admin.html
# open http://localhost:8000/static/register.html
```

### Contents
- `schema.sql` - all tables (core + optional segments/topics).
- `requirements.txt` - Python dependencies.
- `src/music_app_system.py` - core class: register/login, membership, logging, analytics, text hooks, segmentation write-back.
- `src/text_mining.py` - lightweight keyword/MBTI/segment rules (placeholder logic).
- `scripts/init_db.py` - create `data/music.db`.
- `scripts/simulate_data.py` - simulate ~800 users, preferences, 5-200 listens/user, 0-3 feedbacks/user, and classify segments.
- `scripts/demo.py` - end-to-end flow for a new user.
- `scripts/generate_dashboard.py` - produces DAU, registration trend, preference, member-vs-non-member, segment charts -> `outputs/`.
- `static/admin.html` - static dashboard (shows PNGs from `outputs/`).
- `static/register.html` - static user registration form (front-end only; wire to backend API).

### Tips / Troubleshooting
- If imports fail, run scripts with `python -m scripts.xxx` from repo root (puts `src` on path).
- Timestamps may include fractional seconds; dashboard parsing uses `errors="coerce"`.
- Regenerate data any time: rerun `init_db.py` (if you want a clean DB) then `-m scripts.simulate_data`.

---

## 中文版本
请点击上方按钮或链接查看 `README_zh.md`。
