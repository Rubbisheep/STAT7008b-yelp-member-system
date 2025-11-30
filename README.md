# Music Membership Analytics

[English](README.md) | [中文](README_zh.md)

End-to-end toy pipeline for a music app: create schema → simulate user/behavior data → light text-mining hooks → dashboards → static pages + FastAPI.

Admin and registration pages:
![Admin Page](pics/admin.png)
![Register Page](pics/register.png)

## Setup
- Python 3.9+ (tested on Windows; SQLite only)
- From repo root:
  ```bash
  python -m venv .venv
  .venv/Scripts/activate
  pip install -r requirements.txt
  ```

## Run the pipeline (repo root)
1) Optional demo (register a new user, log actions, enrich profile):  
`python -m scripts.demo`

2) Build base charts to `outputs/`:  
`python -m scripts.generate_dashboard`

3) (New) Run advanced analysis to `outputs/advanced/`:  
`python -m scripts.advanced_analysis`

4) Start the API and open the pages:  
`uvicorn api.server:app --reload --port 8001`  
Open `http://localhost:8001/static/admin.html` (dashboard/actions) and `http://localhost:8001/static/register.html` (user sign-up).

## Project map
- `schema.sql` — table definitions (core + segments/topics)
- `src/music_app_system.py` — main class for registration, membership, logging, analytics hooks, segmentation
- `src/text_mining.py` — placeholder keyword/MBTI/segment rules
- `scripts/init_db.py` — build the SQLite DB
- `scripts/simulate_data.py` — refresh with random users, preferences, listening logs, feedback
- `scripts/demo.py` — small end-to-end example for a single user
- `scripts/generate_dashboard.py` — DAU, growth, segment pie, activity heatmap, retention PNGs
- `scripts/advanced_analysis.py` — deeper analysis (behavior patterns, preferences, retention, clustering, churn model, genre similarity) to `outputs/advanced/`
- `api/server.py` — FastAPI backend for register/search/actions (init, simulate, dashboard)
- `static/admin.html` — admin console (PNG display, user lookup, action buttons with status)
- `static/register.html` — user registration form
