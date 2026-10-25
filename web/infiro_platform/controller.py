"""Scope the existing optimizer controller to a study in the one application DB.

The optimizer controller intentionally has global reconciliation and table names.
An identifier-only SQLite adapter gives each study its own tables without copying
its lifecycle logic or creating additional durable databases. The namespace is a
database UUID, never user-provided SQL. This also scopes legacy all-run reporting.
"""

import fcntl
import importlib
import re
import sqlite3
from contextlib import contextmanager

from django.conf import settings
from llm_optimizer.remote.adapters import NebiusAdapter, S3Artifacts
from llm_optimizer.remote.controller import Controller

from .security import private_dir


class ScopedConnection(sqlite3.Connection):
    def sql(self, statement):
        return re.sub(r"\b(runs|jobs)\b", lambda m: self.namespace + m[0], statement)

    def execute(self, statement, parameters=(), /):
        return super().execute(self.sql(statement), parameters)

    def executemany(self, statement, parameters, /):
        return super().executemany(self.sql(statement), parameters)

    def executescript(self, statement, /):
        return super().executescript(self.sql(statement))


class DryRunProvider(NebiusAdapter):
    def submit(self, payload):
        if not payload["request"]["dry_run"]:
            raise ValueError("Live execution requires a credential-safe provider adapter.")
        return super().submit(payload)


class ScopedController(Controller):
    def __init__(self, job, provider=None, s3_factory=S3Artifacts):
        self.namespace = "infiro_ctl_" + job.id.hex + "_"
        self.lease_path = private_dir(job.owner_id, job.id) / "controller.lock"
        super().__init__(
            settings.DATABASES["default"]["NAME"],
            nebius=provider or DryRunProvider(),
            s3_factory=s3_factory,
            concurrency=3,
        )

    def _db(self):
        db = sqlite3.connect(
            self.database,
            timeout=30,
            factory=ScopedConnection,
            uri=self.database.startswith("file:"),
        )
        db.namespace = self.namespace
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def exclusive(self):
        # Preserve the controller's locking contract, but scope its sidecar to
        # this job rather than the shared database (also works for test URIs).
        with self._thread_lock:
            if self._owns_lease:
                yield
                return
            with self.lease_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._owns_lease = True
                try:
                    yield
                finally:
                    self._owns_lease = False
                    fcntl.flock(lock, fcntl.LOCK_UN)


def make_controller(job, prepared):
    if prepared.choices["dry_run"]:
        return ScopedController(job)
    if not settings.INFIRO_LIVE_ENABLED or not settings.INFIRO_PROVIDER_FACTORY:
        raise ValueError(
            "Live execution is disabled until secure credential delivery "
            "and worker images are configured."
        )
    module, name = settings.INFIRO_PROVIDER_FACTORY.rsplit(":", 1)
    # Operator-configured Python dependency injection, never a browser input.
    provider, s3_factory = getattr(importlib.import_module(module), name)(job, prepared)
    if isinstance(provider, NebiusAdapter):
        raise ValueError("The legacy command adapter cannot be used for private web jobs.")
    return ScopedController(job, provider=provider, s3_factory=s3_factory)
