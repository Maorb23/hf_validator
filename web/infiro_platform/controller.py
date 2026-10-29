"""Scope the pinned optimizer controller to a study in the application database.

The optimizer has a small SQLite connection contract. PostgreSQL translates that
contract at this boundary, retaining the shared lifecycle and per-study tables.
Namespaces come from database UUIDs, never user-provided SQL.
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


class IndexedRow(dict):
    """The optimizer reads rows both by column name and by numeric position."""

    def __getitem__(self, key):
        return list(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


def indexed_row(cursor):
    columns = [column.name for column in cursor.description] if cursor.description else []
    return lambda values: IndexedRow(zip(columns, values, strict=True))


class PostgresScopedConnection:
    """Connection adapter for the SQL emitted by the pinned optimizer controller."""

    def __init__(self, namespace):
        import psycopg
        from django.db import connections

        self.namespace = namespace
        parameters = connections["default"].get_connection_params()
        parameters["row_factory"] = indexed_row
        self.connection = psycopg.connect(**parameters)

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, *exc):
        return self.connection.__exit__(*exc)

    def sql(self, statement):
        statement = re.sub(r"\b(runs|jobs)\b", lambda m: self.namespace + m[0], statement)
        return statement.replace("?", "%s")

    def execute(self, statement, parameters=(), /):
        if statement.strip() == "BEGIN IMMEDIATE":
            # Preserve the optimizer's serialized submit/idempotency check. A
            # transaction-scoped table lock also covers other processes.
            return self.connection.execute(
                self.sql("LOCK TABLE runs, jobs IN EXCLUSIVE MODE")
            )
        # The optimizer probes one legacy column at initialization. Translate its
        # SQLite metadata lookup rather than copying its schema/lifecycle logic.
        if statement.strip() == "PRAGMA table_info(jobs)":
            return self.connection.execute(
                "SELECT ordinal_position - 1 AS cid, column_name AS name "
                "FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s "
                "ORDER BY ordinal_position",
                (self.namespace + "jobs",),
            )
        return self.connection.execute(self.sql(statement), parameters)

    def executemany(self, statement, parameters, /):
        cursor = self.connection.cursor()
        cursor.executemany(self.sql(statement), parameters)
        return cursor

    def executescript(self, statement, /):
        # Only the pinned controller's static CREATE TABLE statements use this.
        for command in statement.split(";"):
            if command.strip():
                self.execute(command)


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
        if settings.DATABASES["default"]["ENGINE"] == "django.db.backends.postgresql":
            return PostgresScopedConnection(self.namespace)
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
