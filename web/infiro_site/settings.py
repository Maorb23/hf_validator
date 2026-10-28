"""Public Astro pages and the private, session-authenticated Infiro platform."""
import os
import secrets
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent
if os.environ.get("INFIRO_ENV_FILE"):
    from dotenv import load_dotenv
    load_dotenv(os.environ["INFIRO_ENV_FILE"], override=False)
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
    # Public-only previews may use an ephemeral key. The private platform's
    # local environment helper configures a persistent session signing key.
    SECRET_KEY = SECRET_KEY or secrets.token_urlsafe(50)

INSTALLED_APPS = ["django.contrib.auth", "django.contrib.contenttypes", "django.contrib.sessions", "infiro_site.apps.InfiroSiteConfig", "infiro_platform"]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": os.environ.get("INFIRO_DATABASE", str(BASE_DIR / "private" / "platform.sqlite3")), "OPTIONS": {"timeout": 30, "transaction_mode": "IMMEDIATE"}}}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
TEMPLATES = [{"BACKEND": "django.template.backends.django.DjangoTemplates", "DIRS": [], "APP_DIRS": True, "OPTIONS": {"context_processors": ["django.template.context_processors.request", "django.contrib.auth.context_processors.auth"]}}]
INFIRO_PRIVATE_ROOT = Path(os.environ.get("INFIRO_PRIVATE_ROOT", BASE_DIR / "private")).resolve()
INFIRO_PRIVATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
INFIRO_CREDENTIAL_KEYS = os.environ.get("INFIRO_CREDENTIAL_KEYS", "")
INFIRO_PROJECT_ROOT = BASE_DIR.parent
INFIRO_PROVIDER_FACTORY = os.environ.get("INFIRO_PROVIDER_FACTORY", "infiro_platform.nebius_provider:factory")
import json
INFIRO_SERVING_IMAGES = json.loads(
    os.environ["INFIRO_SERVING_IMAGES"]
    if "INFIRO_SERVING_IMAGES" in os.environ
    else (INFIRO_PROJECT_ROOT / "config" / "platform-images.json").read_text()
)
INFIRO_LIVE_ENABLED = os.environ.get("INFIRO_LIVE_ENABLED") == "1"
INFIRO_WANDB = json.loads(os.environ.get("INFIRO_WANDB", "{}"))
INFIRO_IMAGE_ALIASES = json.loads(
    os.environ["INFIRO_IMAGE_ALIASES"]
    if "INFIRO_IMAGE_ALIASES" in os.environ
    else (INFIRO_PROJECT_ROOT / "config" / "platform-image-aliases.json").read_text()
)
LOGIN_URL = "/app/login"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_AGE = 28800
SESSION_SAVE_EVERY_REQUEST = False
DATA_UPLOAD_MAX_MEMORY_SIZE = 131072
AUTH_PASSWORD_VALIDATORS = [{"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 12}}, {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"}]
ROOT_URLCONF = "infiro_site.urls"
WSGI_APPLICATION = "infiro_site.wsgi.application"
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "infiro_site.middleware.ValidateHostMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "infiro_platform.middleware.PrivateAssetMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "infiro_platform.middleware.PrivateResponseMiddleware",
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

# Railway probes the container over HTTP; all user routes still require HTTPS.
SECURE_REDIRECT_EXEMPT = [r"^healthz$"]

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
