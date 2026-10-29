from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from .database import database_config, database_url


class DatabaseConfigurationTests(SimpleTestCase):
    def test_postgres_url_takes_precedence_and_round_trips_encoded_credentials(self):
        config = database_config(
            "postgresql://test%40user:private%3A%2F%40value@[::1]:5432/platform?sslmode=require",
            "/unused/platform.sqlite3",
        )
        self.assertEqual(config["ENGINE"], "django.db.backends.postgresql")
        self.assertEqual(config["USER"], "test@user")
        self.assertEqual(config["PASSWORD"], "private:/@value")
        self.assertEqual(config["OPTIONS"], {"sslmode": "require", "connect_timeout": 10})
        self.assertNotIn("transaction_mode", config["OPTIONS"])
        self.assertEqual(database_config(database_url(config), "unused"), config)

    def test_sqlite_is_only_the_explicit_no_url_fallback(self):
        config = database_config(None, "/local/private/platform.sqlite3")
        self.assertEqual(config["ENGINE"], "django.db.backends.sqlite3")
        self.assertEqual(config["OPTIONS"]["transaction_mode"], "IMMEDIATE")

    def test_invalid_or_unsupported_urls_fail_without_exposing_credentials(self):
        for value in (
            "",
            "private-password",
            "mysql://user:private@host/db",
            "postgres://user:private@host:invalid/db",
            "postgres://host/",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ImproperlyConfigured) as error:
                    database_config(value, "unused")
                self.assertEqual(
                    str(error.exception), "DATABASE_URL must be a valid PostgreSQL connection URL."
                )
