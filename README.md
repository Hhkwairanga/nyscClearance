NYSC Clearance — Full‑Stack App

Overview
- A Django + React application for managing NYSC corpers across organizations and branch offices. Includes email verification, role‑based dashboards (Organization, Branch Admin, Corper), notifications, leave approvals, and public holidays.

Project Structure
- `nyscClearance/`
  - `nyscClearance_backend/` — Django REST API (accounts app)
  - `nyscClearance_frontend/` — React (Vite) frontend

Backend (Django)
- Python: 3.10+
- Setup
  - `cd nyscClearance/nyscClearance_backend`
  - Create venv (optional) and install: `pip install -r requirements.txt`
  - Set env vars as needed (email backend, DB, CORS, `FRONTEND_ORIGIN`)
  - For local dev, SQLite fallback is enabled (`USE_SQLITE=true`)
  - Migrate: `python manage.py migrate`
  - Run: `python manage.py runserver 0.0.0.0:8000`
- Key endpoints (prefix `/api/auth/`)
  - Auth: `register/`, `verify/`, `csrf/`, `login/`, `logout/`, `me/`
  - Profile: `profile/` (org profile incl. logo, lat/lng)
  - Structure: `branches/`, `departments/`, `units/`
  - Corpers: `corpers/` (enroll requires branch; dept/unit optional)
  - Stats: `stats/`
  - Holidays: `holidays/` (org creates; fields: title, start_date, end_date)
  - Leaves: `leaves/` (corpers create; branch/org approve/reject)
  - Notifications: `notifications/` (org → all/branch; branch → own branch)

Frontend (React + Vite)
- Node 18+
- Setup
  - `cd nyscClearance/nyscClearance_frontend`
  - `npm install`
  - Dev: `npm run dev` (http://localhost:5173)
- Pages
  - `/login` (chooser), `/login/org`, `/login/branch`, `/login/corper`
  - `/signup`, `/verify-success`, `/forgot-password`, `/reset-password`
  - `/dashboard` (role‑based nav)

Core Flows
- Organization signup → email verify → login
- Branch creation (with admin invite) → admin verify → password set → login
- Corper enrollment (requires branch) → verify → password set → login
- Leave management: corper applies; branch approves/rejects
- Notifications: org to all/branch; branch to own branch; corper sees notifications
- Holidays: org creates start_date/end_date; frontend lists and allows delete

Environment & Security
- Use env vars for secrets and email
- CORS/CSRF configured for dev with `FRONTEND_ORIGIN`
- Media (logos) served in dev; use proper storage in prod

Development Notes
- Migrations live in `nyscClearance_backend/accounts/migrations/`
- SQLite fallback for local; production should use Postgres (set `USE_SQLITE=false` + DB vars)

