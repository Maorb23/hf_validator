# Django serving and rollback

## Design

Astro remains the source renderer and React remains the interactive frontend. Django serves the exact built HTML; WhiteNoise serves the same root-relative `/_astro/`, `/data/`, `/logos/`, and download URLs. There is no template conversion, Node proxy, inference import, database, authentication, or schema migration.

`web/infiro_site/` is an independent Django project with CPU-only dependencies in `web/requirements-django.txt`. Django 5.2 LTS and its serving dependencies are pinned. Existing CLI and inference workflows are untouched.

Pretty page paths work with or without a trailing slash and preserve query strings. Unknown paths and missing assets return the original custom 404 with status 404; they never return the homepage as a fallback. Hosts are validated before static middleware handles requests.

## Local use

From the repository root:

```bash
cd web
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-django.txt
npm ci
npm run build
python manage.py check
python manage.py runserver 127.0.0.1:8000
```

Open http://localhost:8000. No `migrate` or `collectstatic` step is required. A second manifest-hashing pipeline would change Astro's URLs unnecessarily. Rebuild and restart the serving process after frontend changes: WhiteNoise indexes assets at startup. `manage.py check` validates page and island references before serving.

## Production-ready serving

No public deployment has been performed or authorized. Prepare a deployment as follows:

1. Build a complete artifact in `web/dist/`, or set `INFIRO_FRONTEND_DIST` to an absolute immutable release directory.
2. Set `INFIRO_ENV=production`, `DJANGO_SECRET_KEY` to a securely generated secret of at least 50 characters, and `DJANGO_ALLOWED_HOSTS` to the approved comma-separated hostnames. Production rejects a missing key, missing host configuration, or wildcard hosts. Do not commit secrets.
3. Set `SITE_URL` to the approved origin when building so canonical links and the sitemap use that domain.
4. With the Python environment activated, run from `web/`:

```bash
python manage.py check --deploy
gunicorn infiro_site.wsgi:application --bind 127.0.0.1:8000 --workers 2 --access-logfile - --error-logfile -
```

Terminate TLS at a trusted reverse proxy. HTTPS redirect and HSTS are enabled in production mode. Set `DJANGO_TRUST_PROXY_SSL_HEADER=1` only when that proxy strips untrusted forwarded headers and supplies `X-Forwarded-Proto`. `DJANGO_SECURE_SSL_REDIRECT=0` is an explicit override for environments handling redirects elsewhere. Local mode allows loopback hosts, needs no secret setup, and still uses DEBUG=False.

`GET /healthz` returns 200 when the frontend entrypoint and asset directory exist, otherwise 503. The full build-reference validation is `manage.py check`. No provider/controller process is inspected or started.

Hashed `/_astro/` assets receive immutable caching. Unversioned data/downloads remain revalidatable with ETags; HTML pages use no-cache. Reuse complete artifacts without renaming assets. Retain older hashed assets for in-flight pages during a zero-downtime release switch.

## Verification

```bash
python manage.py test infiro_site
INFIRO_URL=http://127.0.0.1:8000 PLAYWRIGHT_BROWSERS_PATH=/tmp/infiro-playwright npm run check:browser
```

Tests cover exact page bytes, routes, MIME types, HEAD/range/conditional static requests, downloads, caching, host validation, path traversal, build readiness, and 404s. The existing browser suite runs all 11 pages and demo interactions at desktop/mobile sizes against Django.

## Rollback

`npm run preview` still serves the same artifact on port 4321. Source editing remains `npm run dev`. Revert routing to the static server or restart Django against a retained artifact using `INFIRO_FRONTEND_DIST`. No database or worker state needs rollback.

The Infiro name is provisional. The social image and technology marks are local assets. The conversion is the evaluation-guide download; a technical-discussion link requires a supplied destination.

References: [Django/Gunicorn](https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/gunicorn/), [WhiteNoise](https://whitenoise.readthedocs.io/en/stable/django.html), [Django deployment checklist](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/).
