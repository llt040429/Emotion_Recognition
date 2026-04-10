# System Setup (Vue3 + FastAPI + SQLite)

## 1) Install dependencies

Backend environment:

```bash
pip install -r requirements_system.txt
```

Frontend environment:

```bash
cd frontend
npm install
```

## 2) Model paths

By default backend loads emotion model from:

- `models/best_model`

By default backend loads Silero VAD from:

- `models/silero_vad/silero_vad.jit`

Set a custom path with env var:

```bash
export EMOTION_MODEL_DIR=/absolute/path/to/best_model
export SILERO_VAD_PATH=/absolute/path/to/silero_vad.jit
```

## 3) Run backend

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## 4) Run frontend

```bash
cd frontend
npm run dev
```

Note: Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`, so local development works without CORS setup.

If backend is not on `http://127.0.0.1:8000`, set:

```bash
export VITE_API_BASE=http://your-host:8000
```

