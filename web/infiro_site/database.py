"""Runtime database configuration; malformed URLs never fall back to SQLite."""

from django.core.exceptions import ImproperlyConfigured


def database_config(url, sqlite_path, *, production=False):
    if url is None:
        if production:
            raise ImproperlyConfigured("Production requires a persistent PostgreSQL DATABASE_URL.")
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": str(sqlite_path),
            "OPTIONS": {"timeout": 30, "transaction_mode": "IMMEDIATE"},
        }
    # Parse through the standard adapter without allowing its exception to expose
    # a credential-bearing URL in startup logs.
    try:
        import dj_database_url

        config = dj_database_url.parse(url, conn_max_age=60, conn_health_checks=True)
        if (
            config["ENGINE"] != "django.db.backends.postgresql"
            or not config.get("NAME")
            or not config.get("HOST")
        ):
            raise ValueError
        config.setdefault("OPTIONS", {}).setdefault("connect_timeout", 10)
        return config
    except Exception:
        raise ImproperlyConfigured(
            "DATABASE_URL must be a valid PostgreSQL connection URL."
        ) from None


def database_url(config):
    """Serialize effective PostgreSQL settings for an isolated worker child."""
    from urllib.parse import quote, urlencode

    host = config["HOST"]
    if ":" in host:
        host = f"[{host}]"
    port = f":{config['PORT']}" if config.get("PORT") else ""
    user = quote(config.get("USER", ""), safe="")
    password = quote(config.get("PASSWORD", ""), safe="")
    name = quote(config["NAME"], safe="")
    options = urlencode(config.get("OPTIONS", {}))
    return f"postgresql://{user}:{password}@{host}{port}/{name}" + (
        f"?{options}" if options else ""
    )
