"""Offline companion to Django loaddata when moving a stopped SQLite installation."""

import re
import sqlite3
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from infiro_platform.controller import PostgresScopedConnection, ScopedController
from infiro_platform.models import Job

TABLE = re.compile(r"infiro_ctl_([0-9a-f]{32})_(runs|jobs)\Z")
COLUMNS = {
    "runs": ("id", "idempotency_key", "request", "created_at", "cancelled"),
    "jobs": ("run_id", "framework", "state", "nebius_id", "error", "progress", "updated_at"),
}


class Command(BaseCommand):
    help = "Import optimizer tables from a stopped SQLite backup into PostgreSQL after loaddata."

    def add_arguments(self, parser):
        parser.add_argument("--source", type=Path, required=True)

    def handle(self, *args, **options):
        if settings.DATABASES["default"]["ENGINE"] != "django.db.backends.postgresql":
            raise CommandError("The target must be PostgreSQL.")
        if settings.INFIRO_LIVE_ENABLED:
            raise CommandError(
                "Disable live execution and stop all writers/workers before importing."
            )
        path = Path(options["source"]).resolve()
        if not path.is_file():
            raise CommandError("Source SQLite backup is missing.")
        source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        source.row_factory = sqlite3.Row
        try:
            names = [
                row[0]
                for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            tables = {}
            for name in names:
                match = TABLE.fullmatch(name)
                if match:
                    tables.setdefault(match[1], set()).add(match[2])
            expected = {
                identifier.hex
                for identifier in Job.objects.filter(kind="study")
                .exclude(controller_run="")
                .values_list("id", flat=True)
            }
            if not expected.issubset(tables):
                raise CommandError(
                    "Source backup is missing controller tables for imported studies."
                )
            # Validate the whole source before creating any target tables.
            for identifier, kinds in tables.items():
                if kinds != set(COLUMNS) or not Job.objects.filter(pk=identifier).exists():
                    raise CommandError(
                        "Import the matching Django fixture first; controller owners must exist."
                    )
            for identifier in tables:
                job = Job.objects.get(pk=identifier)
                controller = ScopedController(job)
                with PostgresScopedConnection(controller.namespace) as target:
                    target.execute("BEGIN IMMEDIATE")
                    if any(
                        target.execute(f"SELECT COUNT(*) FROM {kind}").fetchone()[0]
                        for kind in COLUMNS
                    ):
                        raise CommandError(
                            "Target controller already contains records; refusing to overwrite."
                        )
                    for kind, columns in COLUMNS.items():
                        rows = source.execute(
                            f'SELECT * FROM "infiro_ctl_{identifier}_{kind}"'
                        ).fetchall()
                        values = []
                        for row in rows:
                            # Old optimizer schemas may predate progress storage.
                            if any(
                                column not in row.keys()
                                for column in columns
                                if column != "progress"
                            ):
                                raise CommandError("Unsupported source controller schema.")
                            values.append(
                                tuple(
                                    row[column] if column in row.keys() else None
                                    for column in columns
                                )
                            )
                        target.executemany(
                            f"INSERT INTO {kind} ({', '.join(columns)}) VALUES "
                            f"({', '.join('?' for _ in columns)})",
                            values,
                        )
                # Apply the optimizer's legacy error redaction to the imported state.
                ScopedController(job)
                self.stdout.write("Imported one controller; source backup remains unchanged.")
        finally:
            source.close()
