"""Django hosts the existing static frontend; no database or auth is configured."""
import os
import secrets
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = Path(os.environ.get("INFIRO_FRONTEND_DIST", BASE_DIR / "dist")).resolve()
ENVIRONMENT = os.environ.get("INFIRO_ENV", "local")
if ENVIRONMENT not in {"local", "production"}:
    raise ImproperlyConfigured("INFIRO_ENV must be local or production")

DEBUG = False
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if h.strip()]
if ENVIRONMENT == "production":
    if len(SECRET_KEY) < 50:
        raise ImproperlyConfigured("Production requires DJANGO_SECRET_KEY with at least 50 characters")
    if not os.environ.get("DJANGO_ALLOWED_HOSTS") or not ALLOWED_HOSTS or "*" in ALLOWED_HOSTS:
        raise ImproperlyConfigured("Production requires explicit DJANGO_ALLOWED_HOSTS without a wildcard")
else:
    # No sessions/signatures are persisted. Each local process can use an ephemeral key.
    SECRET_KEY = SECRET_KEY or secrets.token_urlsafe(50)

INSTALLED_APPS = ["infiro_site.apps.InfiroSiteConfig"]
DATABASES = {}
ROOT_URLCONF = "infiro_site.urls"
WSGI_APPLICATION = "infiro_site.wsgi.application"
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "infiro_site.middleware.ValidateHostMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]
APPEND_SLASH = False
USE_TZ = True
TIME_ZONE = "UTC"
STATIC_URL = "/static/"
STATIC_ROOT = None

# Astro already fingerprints assets and writes root-relative URLs. Do not rename
# them with collectstatic, rewrite HTML, or move data under /static/.
WHITENOISE_ROOT = FRONTEND_DIST
WHITENOISE_USE_FINDERS = False
WHITENOISE_AUTOREFRESH = False
WHITENOISE_INDEX_FILE = False
WHITENOISE_MAX_AGE = 0
WHITENOISE_IMMUTABLE_FILE_TEST = r"^/_astro/"
WHITENOISE_MIMETYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".md": "text/markdown", ".xml": "application/xml", ".woff2": "font/woff2"}

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"
SESSION_COOKIE_SECURE = ENVIRONMENT == "production"
CSRF_COOKIE_SECURE = ENVIRONMENT == "production"
SECURE_SSL_REDIRECT = os.environ.get("DJANGO_SECURE_SSL_REDIRECT", "1" if ENVIRONMENT == "production" else "0") == "1"
SECURE_HSTS_SECONDS = 31536000 if ENVIRONMENT == "production" else 0
# Trust forwarded HTTPS only when explicitly configured behind a trusted proxy.
if os.environ.get("DJANGO_TRUST_PROXY_SSL_HEADER") == "1":
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
