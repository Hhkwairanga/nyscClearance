import os
from pathlib import Path
from corsheaders.defaults import default_headers
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file early, without external dependencies
def _load_env_file():
    def parse_and_set(path: Path):
        try:
            with path.open('r') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    # Strip optional surrounding quotes
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    # Do not overwrite existing environment variables
                    if key not in os.environ:
                        os.environ[key] = val
        except FileNotFoundError:
            pass

    env_choice = os.getenv('DJANGO_ENV', '').lower()
    # Determine candidate filenames based on environment choice
    filenames = []
    if env_choice in ('production', 'prod'):
        filenames = ['.env', '.env.prod']
    elif env_choice in ('development', 'dev'):
        filenames = ['.env', '.env.local']
    else:
        # Default to dev-style
        filenames = ['.env', '.env.local']

    # Try both backend folder and project root (parent of BASE_DIR)
    search_dirs = [BASE_DIR, BASE_DIR.parent]
    for name in filenames:
        for d in search_dirs:
            parse_and_set(d / name)

_load_env_file()

def _csv_env(name, default_list=None):
    val = os.getenv(name)
    if val is None or not str(val).strip():
        return list(default_list or [])
    return [v.strip() for v in str(val).split(',') if v.strip()]

def _bool_env(name, default=False):
    val = os.getenv(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in ('1','true','yes','on')

# Core envs
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'dev-secret-key-change-me')
DEBUG = os.getenv('DJANGO_DEBUG', 'true').lower() == 'true'
ALLOWED_HOSTS = _csv_env('DJANGO_ALLOWED_HOSTS', ['*'] if DEBUG else [])
# If still empty in development (e.g., empty env var), use standard dev hosts
if DEBUG and not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ['localhost', '127.0.0.1', '[::1]']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'accounts',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'nysc_clearance.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'nysc_clearance.wsgi.application'


# Always use PostgreSQL (no SQLite fallback)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('POSTGRES_DB', 'nyscClearance_db'),
        # Do not hardcode credentials; use env with safe defaults
        'USER': os.getenv('POSTGRES_USER'),
        'PASSWORD': os.getenv('POSTGRES_PASSWORD'),
        'HOST': os.getenv('POSTGRES_HOST', 'localhost'),
        'PORT': os.getenv('POSTGRES_PORT', '5432'),
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


LANGUAGE_CODE = 'en-us'
# Set your local server time zone to avoid admin offset message
TIME_ZONE = os.getenv('DJANGO_TIME_ZONE', 'Africa/Lagos')
USE_I18N = True
USE_TZ = True


STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

AUTH_USER_MODEL = 'accounts.OrganizationUser'

EMAIL_BACKEND = os.getenv('DJANGO_EMAIL_BACKEND', 'django.core.mail.backends.smtp.EmailBackend')
# SMTP defaults (production can override via env)
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.hostinger.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true'
EMAIL_USE_SSL = os.getenv('EMAIL_USE_SSL', 'false').lower() == 'true'
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = os.getenv('DJANGO_DEFAULT_FROM_EMAIL', EMAIL_HOST_USER)

# In development, prefer console email backend unless explicitly overridden
if DEBUG and os.getenv('DJANGO_EMAIL_BACKEND') is None:
    EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# CORS / CSRF
# Primary frontend URL and API base, plus optional multiple origins for local DX
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:5173').rstrip('/')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000').rstrip('/')

FRONTEND_ORIGINS = _csv_env('FRONTEND_ORIGINS', [
    FRONTEND_URL,
    'http://localhost:5174',
    'http://127.0.0.1:5173',
    'http://127.0.0.1:5174',
])

CORS_ALLOWED_ORIGINS = FRONTEND_ORIGINS

# CSRF trusted origins come from env (comma-separated). If not provided, trust FRONTEND_URL and
# local dev variants. In production, set DJANGO_CSRF_TRUSTED_ORIGINS explicitly with https:// URLs.
CSRF_TRUSTED_ORIGINS = _csv_env('DJANGO_CSRF_TRUSTED_ORIGINS', FRONTEND_ORIGINS)
CORS_ALLOW_HEADERS = list(default_headers) + [
    'x-csrftoken',
]
CORS_ALLOW_CREDENTIALS = True

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'accounts.auth.SignedTokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}

# Expose frontend origin as a setting for redirects (single URL)
# Prefer explicit FRONTEND_URL; keep legacy FRONTEND_ORIGIN env for compatibility
FRONTEND_ORIGIN = os.getenv('FRONTEND_ORIGIN', FRONTEND_URL)

# Ensure API host is allowed (helps local dev when env vars conflict)
try:
    api_host = urlparse(API_BASE_URL).hostname
    if api_host and api_host not in ALLOWED_HOSTS and ALLOWED_HOSTS != ['*']:
        ALLOWED_HOSTS.append(api_host)
except Exception:
    pass

# In dev, make sure localhost forms are accepted
if DEBUG and ALLOWED_HOSTS != ['*']:
    for h in ('localhost', '127.0.0.1', '[::1]'):
        if h not in ALLOWED_HOSTS:
            ALLOWED_HOSTS.append(h)

# Media uploads (e.g., organization logos)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Cookie and session security (env-driven)
CSRF_COOKIE_NAME = os.getenv('DJANGO_CSRF_COOKIE_NAME', 'csrftoken')
SESSION_COOKIE_NAME = os.getenv('DJANGO_SESSION_COOKIE_NAME', 'sessionid')
SESSION_COOKIE_AGE = int(os.getenv('DJANGO_SESSION_COOKIE_AGE', '1209600'))  # 2 weeks
SESSION_SAVE_EVERY_REQUEST = _bool_env('DJANGO_SESSION_SAVE_EVERY_REQUEST', False)

# SameSite policy. For cross-site SPA (frontend on different subdomain), use 'None' in production.
SESSION_COOKIE_SAMESITE = os.getenv('DJANGO_SESSION_COOKIE_SAMESITE', 'Lax')
CSRF_COOKIE_SAMESITE = os.getenv('DJANGO_CSRF_COOKIE_SAMESITE', SESSION_COOKIE_SAMESITE)

# HttpOnly: JS should not access session cookie; CSRF cookie can be read by JS to set header
SESSION_COOKIE_HTTPONLY = _bool_env('DJANGO_SESSION_COOKIE_HTTPONLY', True)
CSRF_COOKIE_HTTPONLY = _bool_env('DJANGO_CSRF_COOKIE_HTTPONLY', False)

# Secure cookies: required when SameSite=None (browsers enforce)
SESSION_COOKIE_SECURE = _bool_env('DJANGO_SESSION_COOKIE_SECURE', not DEBUG)
CSRF_COOKIE_SECURE = _bool_env('DJANGO_CSRF_COOKIE_SECURE', not DEBUG)

# Optional cookie domains (typically not required). Set if you need a parent domain like .sahabs.tech
SESSION_COOKIE_DOMAIN = os.getenv('DJANGO_SESSION_COOKIE_DOMAIN') or None
CSRF_COOKIE_DOMAIN = os.getenv('DJANGO_CSRF_COOKIE_DOMAIN') or None

# Security headers / HTTPS
SECURE_SSL_REDIRECT = _bool_env('DJANGO_SECURE_SSL_REDIRECT', not DEBUG)
_proxy_hdr = os.getenv('DJANGO_SECURE_PROXY_SSL_HEADER')
if _proxy_hdr:
    try:
        name, val = [s.strip() for s in _proxy_hdr.split(',', 1)]
        SECURE_PROXY_SSL_HEADER = (name, val)
    except Exception:
        SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
elif not DEBUG:
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

SECURE_HSTS_SECONDS = int(os.getenv('DJANGO_SECURE_HSTS_SECONDS', '0' if DEBUG else '31536000'))
SECURE_HSTS_INCLUDE_SUBDOMAINS = _bool_env('DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS', not DEBUG)
SECURE_HSTS_PRELOAD = _bool_env('DJANGO_SECURE_HSTS_PRELOAD', not DEBUG)
SECURE_CONTENT_TYPE_NOSNIFF = _bool_env('DJANGO_SECURE_CONTENT_TYPE_NOSNIFF', True)

# Attendance geofence radius (meters); strict default
ATTENDANCE_GEOFENCE_METERS = int(os.getenv('ATTENDANCE_GEOFENCE_METERS', '100'))
