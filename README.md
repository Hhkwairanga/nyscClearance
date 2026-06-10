NYSC Clearance Manager
======================

A Django + React application to manage organizations, branches, corp members, attendance, and monthly performance clearances with wallet-based billing and Paystack funding.

Overview
--------

- Roles: Organization (ORG), Branch Admin (BRANCH), Corper (CORPER)
- Features:
  - Structure: branches, departments, units, corp member enrollment
  - Attendance: geofenced self-check-in with face detection assist
  - Clearance: generate printable monthly performance clearance letters
  - Wallets: ORG/BRANCH/CORPER wallets with transaction logs
  - Billing: auto-deduct clearance fee; discount and fee configurable
  - Funding: Paystack initialize/verify using keys stored in DB
  - Announcements: admin-configured floating notice for ORG dashboards

Architecture
------------

- Backend: Django REST Framework app in `nyscClearance_backend`
  - App: `accounts` holds models, views, admin
- Frontend: React app in `nyscClearance_frontend`

Setup
-----

1) Backend
- Python >= 3.11, PostgreSQL running
- Env vars (example):
  - `POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT`
  - `DJANGO_SECRET_KEY`
  - `DJANGO_DEBUG=true|false`
  - `DJANGO_ALLOWED_HOSTS=example.com,localhost`
  - `DJANGO_TIME_ZONE=Africa/Lagos` (default provided)
  - Optional email SMTP vars for notifications
- Install deps, run migrations:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python manage.py migrate
    python manage.py createsuperuser
    python manage.py runserver

2) Frontend
- Node 18+

    cd nyscClearance_frontend
    npm install
    npm run dev

- Configure allowed origins via env: `FRONTEND_ORIGINS` in backend settings.

Configuration (URLs)
--------------------

- Frontend → Backend API base:
  - Production on `nyscclearance.com` automatically uses `https://api.nyscclearance.com`.
  - Temporary legacy support keeps `https://nyscclearance.sahabs.tech` pointed at `https://api.sahabs.tech`.
  - Set `VITE_API_BASE` only if you need to override this default.
  - Leave it empty in local dev to use the Vite proxy.
  - Set `VITE_ROOT_DOMAIN=nyscclearance.com` when building for the new primary domain.
  - Optional legacy overrides: `VITE_LEGACY_FRONTEND_HOST=nyscclearance.sahabs.tech`, `VITE_LEGACY_API_BASE=https://api.sahabs.tech`.
  - All API calls and anchor links use helpers in `src/api/urls.js` to resolve against this base.
  - Set `VITE_APP_VERSION` for each production release. If omitted, the Vite build timestamp is used.

- Backend → Frontend origins:
  - Production defaults trust `https://nyscclearance.com`, `https://www.nyscclearance.com`, and wildcard enterprise subdomains.
  - Legacy defaults also trust `https://nyscclearance.sahabs.tech` and allow `api.sahabs.tech`.
  - `FRONTEND_ORIGINS` (comma-separated) for extra CORS/CSRF allowlist entries.
  - `FRONTEND_ORIGIN` for server-generated links back to the SPA (e.g., email links).
  - `APP_ROOT_DOMAIN=nyscclearance.com` and `APP_API_HOSTNAME=api.nyscclearance.com` control backend host/domain defaults.
  - Optional legacy overrides: `APP_LEGACY_FRONTEND_HOSTNAME=nyscclearance.sahabs.tech`, `APP_LEGACY_API_HOSTNAME=api.sahabs.tech`.

Admin Configuration
-------------------

- System Settings (singleton):
  - Welcome Bonus: credited only to ORG wallet on first creation
  - Clearance Fee: base fee per clearance debit
  - Discount: toggle + percent; applied before VAT
  - Announcement: title/message with optional start/end; shows as floating modal for ORG
- Paystack Config:
  - `public_key`, `secret_key`, `webhook_secret` (optional), `is_active`
  - The latest active config is used by the API

Wallets & Billing
-----------------

- Every user has a wallet; ORG wallet receives welcome bonus once.
- Debits (clearance): attempt in order ORG → BRANCH → CORPER
  - If all insufficient, access to clearance is denied with a prompt to fund wallet.
- Credits: from Paystack verify, credited to the authenticated user’s wallet.
- Transactions: `WalletTransaction` with amount, vat_amount, total_amount, description, reference.

Paystack Funding Flow
---------------------

- Initialize: `POST /api/auth/wallet/paystack/initialize/`
  - Body: `{ email, amount, callback_url? }` (amount in NGN)
  - Returns: `{ authorization_url, reference }`
- Verify: `POST /api/auth/wallet/paystack/verify/`
  - Body: `{ reference }`
  - On success: credits wallet and returns `{ status: 'success', balance }`
- Frontend behavior:
  - Wallet modal asks for amount (commas allowed), calls initialize, redirects to Paystack.
  - After payment, return to `.../dashboard?paystack=1&reference=...`; app verifies and refreshes.

Attendance
----------

- Corper-only self-check-in; requires geolocation within configured radius.
- On each finalize, `time_in` is set if missing; `time_out` is always updated to now (not before `time_in`).
- Holidays block check-in.

Automatic Nigerian Public Holidays
---------------------------------

- National (Nigeria) public holidays are synced from the free Nager.Date API and stored in the database.
- Organisations can still add custom/manual holidays from the dashboard.
- Run yearly sync (e.g. via cron):

    python manage.py sync_national_holidays --country NG --year 2026

Clearance Letters
-----------------

- Corper views the monthly clearance page; server attempts to charge by reference (prevents duplicates).
- If charge cannot be made by ORG/BRANCH/CORPER wallets, renders a payment-required page with a link to open the wallet funding modal.
- Printable and downloadable as PDF; includes QR code with verification URL.

Key Endpoints (abbreviated)
---------------------------

- Auth/Profile: `/api/auth/me/`, `/api/auth/profile/`, `/api/auth/stats/`
- Structure: `/api/auth/branches/`, `/api/auth/departments/`, `/api/auth/units/`, `/api/auth/corpers/`
- Attendance: `/api/auth/attendance/` (HTML), plus authorize/process/finalize
- Clearance: `/api/auth/performance/summary/`, `/api/auth/performance/clearance/` (HTML)
- Wallet:
  - `GET /api/auth/wallet/`
  - `POST /api/auth/wallet/charge_clearance/` (corper-triggered download charge)
  - Paystack: `POST /api/auth/wallet/paystack/initialize/`, `POST /api/auth/wallet/paystack/verify/`
- Announcements: `GET /api/auth/announcement/` (ORG only)

Environment & Timezones
-----------------------

- Server timezone default is `Africa/Lagos` (configured via `DJANGO_TIME_ZONE`).
- DRF responses and admin use timezone-aware datetimes.

Development Notes
-----------------

- Avoid double-charging by passing a stable `reference` per month per corper.
- Discount percent should be 0–100; clearance fee non-negative amounts.
- Wallet totals are calculated client-side for display; admin templates show overall totals.

Deployment
----------

- Set secure `DJANGO_SECRET_KEY`, proper `ALLOWED_HOSTS`, and SMTP credentials if used.
- Create System Settings and Paystack Config in admin.
- Configure CORS/CSRF to allow your frontend domain(s).
- For enterprise subdomains, keep wildcard DNS and TLS enabled for `*.nyscclearance.com`; production settings share cookies on `.nyscclearance.com`.
- During migration, old `sahabs.tech` and new `nyscclearance.com` users may need separate logins because browsers cannot share cookies across different parent domains.
- Run migrations, then optionally force a major deployment logout:

    python manage.py force_logout

  `clearsessions` only removes expired Django sessions; use `force_logout` when you need every active user to sign in again after a major deployment. It clears active session rows and increments the System Settings auth token version so previously issued bearer tokens are rejected.

- Production cookie/security defaults are enabled when `DJANGO_DEBUG=false`:
  - `SESSION_COOKIE_SECURE=true`, `CSRF_COOKIE_SECURE=true`
  - `SESSION_COOKIE_HTTPONLY=true`
  - HSTS, SSL redirect, `X_FRAME_OPTIONS=DENY`, and `Referrer-Policy: same-origin`
  - For cross-subdomain deployments, keep HTTPS and set cookie/domain env vars only when needed.
- The React app performs a deployment refresh on startup:
  - Compares `VITE_APP_VERSION` (or build timestamp fallback) with `localStorage.app_version`.
  - On version change, logs out the server session, clears browser storage, unregisters service workers, saves the new version, and reloads once.
  - Stale API auth (`401`, or auth-related `403`) clears the token and redirects to `/login`.

Nginx SPA caching recommendation:

    location = /index.html {
        add_header Cache-Control "no-store, no-cache, must-revalidate, max-age=0" always;
        try_files $uri =404;
    }

    location /assets/ {
        add_header Cache-Control "public, max-age=31536000, immutable" always;
        try_files $uri =404;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://django_backend;
    }
