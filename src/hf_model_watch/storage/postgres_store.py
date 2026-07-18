from __future__ import annotations


class PostgresStore:
    """Future-compatible SQLAlchemy-backed Postgres store.

    Phase 3 keeps SQLite as the implemented default. This class validates that callers
    provided a Postgres SQLAlchemy URL and defers real implementation to a later phase.
    """

    def __init__(self, sqlalchemy_url: str) -> None:
        if not sqlalchemy_url.startswith(("postgresql://", "postgresql+")):
            raise ValueError("PostgresStore requires a PostgreSQL SQLAlchemy URL")
        self.sqlalchemy_url = sqlalchemy_url

    def initialize(self) -> None:
        try:
            import sqlalchemy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install sqlalchemy to use PostgresStore") from exc
        raise NotImplementedError("PostgresStore is a Phase 3 optional skeleton")
