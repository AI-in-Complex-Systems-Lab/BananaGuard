# BananaGuard Web Platform

A FastAPI + React platform for firearm detection: upload video for
batch processing, watch a live camera feed, and have reviewers
approve, reject, or correct every detection the model makes. Access
is gated by accounts with `admin` or `officer` roles.

## Prerequisites

- Python 3.11+ (3.13 also works)
- Node.js 20+
- [Git LFS](https://git-lfs.com/) — `backend/models/weapon_detection.pt`
  is tracked via LFS (it's 124MB, over GitHub's normal push limit).
  Run `git lfs install && git lfs pull` after cloning, or the backend
  will silently fall back to the smaller, less accurate `yolo11.pt` at
  the repo root.
- A YOLO weights file. By default the backend looks for
  `backend/models/weapon_detection.pt` (falling back to `yolo11.pt` in
  the repository root if that's missing); override with `MODEL_PATH`
  if yours lives elsewhere.

## Backend setup

```bash
cd webapp/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

The server listens on `PORT` (default `8080`; the frontend's default
config expects `8081` — see below). On first run, if no user accounts
exist yet, it creates a default `admin` account with a random
password. That password is:

- printed to the server's console output, and
- written to `storage/admin_bootstrap.txt`, and
- shown directly on the login page until the admin password is changed.

**Change that password before deploying anywhere real.**

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Port the server listens on |
| `MODEL_PATH` | `backend/models/weapon_detection.pt` (falls back to `<repo root>/yolo11.pt`) | Path to the YOLO weights file |
| `CONFIDENCE_THRESHOLD` | `0.50` | Initial detection confidence threshold (admins can change this at runtime from the Settings page) |
| `AUTH_SECRET_KEY` | auto-generated, persisted to `storage/auth_secret.key` | Secret used to sign session tokens. Set this explicitly in production so sessions survive a redeploy that wipes local storage. |
| `STORAGE_DIR` | `backend/storage` | Where uploads, outputs, jobs, reviews, and users are persisted. Mainly useful for pointing tests at a temp directory. |

### Running the backend test suite

```bash
cd webapp/backend
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Tests spin up the real FastAPI app (including loading the YOLO model
once) against a temporary storage directory, so they don't touch or
depend on your local `storage/` data.

## Frontend setup

```bash
cd webapp/frontend
npm install
npm run dev
```

Vite prints the local URL (default `http://localhost:5173`, falling
back to the next free port if that's taken).

### Environment variables

Set these in `webapp/frontend/.env.local` (create the file if it
doesn't exist):

```
VITE_API_URL=http://localhost:8081
VITE_WS_URL=ws://localhost:8081/ws
```

Point them at wherever your backend is actually running.

## Deployment

Both `backend/Dockerfile` and `frontend/Dockerfile` build standalone
images (the frontend one serves the built static files via Nginx,
templated for Cloud Run's `$PORT`). Set `AUTH_SECRET_KEY` explicitly
for the backend container rather than relying on the auto-generated
default, since container filesystems are typically ephemeral. Make
sure `git lfs pull` has run before `docker build` — the model weights
won't be present otherwise.

## Project layout

```
webapp/
  backend/
    server.py        FastAPI app: video upload/processing, jobs, dashboard, frames
    auth.py           JWT session handling
    auth_api.py       Login, user management, settings-adjacent auth endpoints
    user_store.py     Local JSON-backed user accounts (hashed passwords)
    job_store.py      Persists job metadata across restarts
    review_api.py     Detection review endpoints (approve/reject/correct)
    review_store.py   Persists review state per job
    dataset_export.py Builds a YOLO-format dataset zip from approved/corrected reviews
    tests/            pytest suite
  frontend/
    src/
      App.jsx             Top-level routing between views
      AuthContext.jsx     Session state, login/logout
      LoginPage.jsx
      AppShell.jsx        Sidebar + topbar layout
      DashboardPage.jsx
      UploadPanel.jsx
      WebcamPanel.jsx      Live camera feed with auto-reconnect
      JobHistoryPanel.jsx
      ReviewPanel.jsx      Human review table
      BoxCorrectionModal.jsx  Drag-to-redraw box correction on the real frame
      UsersAdminPage.jsx   Admin-only user management
      SettingsPage.jsx     Admin-only confidence threshold control
      theme.css            Dark command-console design system
```
