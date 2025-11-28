# 音乐会员分析

[English](README.md) | [中文](README_zh.md)

端到端示例：建库 → 模拟用户/行为数据 → 简易文本挖掘钩子 → 看板 → 静态页 + FastAPI。

管理端与注册页：
![Admin 页面](pics/admin.png)
![注册页面](pics/register.png)

## 环境准备
- Python 3.9+
- 仓库根目录执行：
  ```bash
  python -m venv .venv
  .venv/Scripts/activate
  pip install -r requirements.txt
  ```

## 跑通流程（仓库根目录）

1）注册新用户、写日志、补全画像：  
`python -m scripts.demo`

2）产出图表到 `outputs/`：  
`python -m scripts.generate_dashboard`

3）启动 API 并打开页面：  
`uvicorn api.server:app --reload --port 8001`  
访问 `http://localhost:8001/static/admin.html`（管理与操作）和 `http://localhost:8001/static/register.html`（用户注册）。

## 目录速览
- `schema.sql` — 表结构（核心 + 分群/主题）
- `src/music_app_system.py` — 注册、会员、日志、统计钩子、分群的核心类
- `src/text_mining.py` — 关键词/MBTI/分群占位规则
- `scripts/init_db.py` — 创建 SQLite 数据库
- `scripts/simulate_data.py` — 刷新随机用户、偏好、听歌日志、反馈
- `scripts/demo.py` — 单用户端到端演示
- `scripts/generate_dashboard.py` — 生成 DAU、增长、分群占比、活跃热力、留存 PNG
- `api/server.py` — FastAPI 后端（注册/搜索/动作：init、simulate、dashboard）
- `static/admin.html` — 管理端（展示 PNG、查用户、触发动作并轮询状态）
- `static/register.html` — 注册页

## 小贴士
- 用模块方式运行脚本（`python -m scripts.xxx`），保证能找到 `src`。
- 需要重置数据时：先 `python -m scripts.init_db`，再 `python -m scripts.simulate_data`。
- 如前端放在其他域/端口，可在 HTML 里设置 `window.API_BASE` 指向实际 API。
