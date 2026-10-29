"""Real isolated worker and controller checks against a local PostgreSQL test DB."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import skipUnless

from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TransactionTestCase, override_settings

from .controller import ScopedController
from .demo import seed_prepared
from .management.commands.study_logs import provider_jobs
from .queue import child_environment, run_once
from .services import approve


@skipUnless(connection.vendor == "postgresql", "PostgreSQL integration checks")
class PostgresWorkerTests(TransactionTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        override = override_settings(
            INFIRO_PRIVATE_ROOT=Path(directory.name),
            INFIRO_CREDENTIAL_KEYS=Fernet.generate_key().decode(),
        )
        override.enable()
        self.addCleanup(override.disable)
        self.user = get_user_model().objects.create_user("postgres-worker")

    def test_real_child_dry_run_and_controller_recovery(self):
        from infiro_site.database import database_config

        prepared = seed_prepared(self.user)
        job = approve(self.user, prepared.id, prepared.digest, True)
        env = child_environment(job)
        child_database = database_config(env["DATABASE_URL"], "unused")
        self.assertEqual(child_database["NAME"], connection.settings_dict["NAME"])
        self.assertNotIn("INFIRO_DATABASE", env)
        self.assertNotIn("INFIRO_CREDENTIAL_KEYS", env)
        self.assertTrue(run_once())
        job.refresh_from_db()
        self.assertEqual(job.state, "dry_run", job.error)
        original = job.controller_run
        self.assertEqual(ScopedController(job).status(original)["state"], "dry_run")
        self.assertTrue(provider_jobs(job))
        self.assertEqual(ScopedController(job).status(original)["run_id"], original)

    def test_sqlite_controller_import_preserves_existing_run_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from django.core.management import call_command
        from django.core.management.base import CommandError

        from .services import perform_study

        with TemporaryDirectory() as directory:
            source = Path(directory) / "old-platform.sqlite3"
            sqlite_settings = SimpleNamespace(
                DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": str(source)}}
            )
            prepared = seed_prepared(self.user)
            job = approve(self.user, prepared.id, prepared.digest, True)
            with patch("infiro_platform.controller.settings", sqlite_settings):
                perform_study(job)
                job.save()
            self.assertEqual(job.state, "dry_run")
            original = job.controller_run
            call_command("import_sqlite_controllers", source=source)
            self.assertEqual(ScopedController(job).status(original)["state"], "dry_run")
            with self.assertRaisesMessage(CommandError, "refusing to overwrite"):
                call_command("import_sqlite_controllers", source=source)
            self.assertEqual(ScopedController(job).status(original)["run_id"], original)

            import sqlite3

            with sqlite3.connect(source) as old:
                old.execute(f'DROP TABLE "infiro_ctl_{job.id.hex}_jobs"')
                old.execute(f'DROP TABLE "infiro_ctl_{job.id.hex}_runs"')
            with self.assertRaisesMessage(CommandError, "missing controller tables"):
                call_command("import_sqlite_controllers", source=source)
            self.assertEqual(ScopedController(job).status(original)["run_id"], original)

    def test_concurrent_controller_submit_reuses_one_idempotent_run(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        from django.db import connections

        prepared = seed_prepared(self.user)
        job = approve(self.user, prepared.id, prepared.digest, True)
        controller = ScopedController(job)
        barrier = Barrier(2)

        def submit(_):
            try:
                barrier.wait(timeout=10)
                return controller.submit(prepared.payload, idempotency_key=str(job.id))["run_id"]
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(submit, range(2)))
        self.assertEqual(runs[0], runs[1])
        controller.tick()
        self.assertEqual(controller.status(runs[0])["state"], "dry_run")
