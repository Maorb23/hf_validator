from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from hf_model_watch.storage.schemas import utc_now_iso

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS models (
        model_id TEXT PRIMARY KEY,
        model_family TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_revisions (
        model_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        model_family TEXT,
        size_band TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (model_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scan_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        model_id TEXT NOT NULL,
        revision TEXT,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        model_id TEXT NOT NULL,
        revision TEXT,
        eligible INTEGER NOT NULL,
        score REAL NOT NULL,
        blocking_reasons_json TEXT NOT NULL,
        score_breakdown_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        revision TEXT,
        model_family TEXT,
        size_band TEXT,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT NOT NULL,
        failure_category TEXT,
        failure_message TEXT,
        selected_backend TEXT,
        selected_dtype TEXT,
        selected_quantization TEXT,
        base_score REAL,
        adapted_score REAL,
        adaptation_lift REAL,
        latency_summary TEXT,
        context_drift_summary TEXT,
        artifact_uri TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        claims_vs_observed_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        selected_backend TEXT,
        selected_dtype TEXT,
        selected_quantization TEXT,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS latency_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        latency_summary TEXT,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS quality_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        base_score REAL,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context_drift_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        context_drift_summary TEXT,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS finetune_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        status TEXT NOT NULL,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS adaptation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        adapted_score REAL,
        adaptation_lift REAL,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        revision TEXT,
        compared_to_model_id TEXT,
        compared_to_revision TEXT,
        payload_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS failures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        model_id TEXT NOT NULL,
        revision TEXT,
        failure_category TEXT NOT NULL,
        failure_message TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        model_id TEXT,
        revision TEXT,
        artifact_type TEXT NOT NULL,
        artifact_uri TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS result_sync_state (
        run_id TEXT PRIMARY KEY,
        source_uri TEXT NOT NULL,
        content_fingerprint TEXT NOT NULL,
        synced_at TEXT NOT NULL
    )
    """,
)


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            conn.commit()

    def insert_scan_result(
        self,
        candidate: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        now = utc_now_iso()
        model_id = str(candidate["model_id"])
        revision = candidate.get("revision")
        with self._connect() as conn:
            self._upsert_model_revision(conn, model_id, revision, candidate)
            conn.execute(
                """
                INSERT INTO scan_results (run_id, model_id, revision, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, _json(candidate), now),
            )
            conn.commit()

    def insert_candidate_score(
        self,
        score_record: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        now = utc_now_iso()
        model_id = str(score_record["model_id"])
        revision = score_record.get("revision")
        with self._connect() as conn:
            self._upsert_model_revision(conn, model_id, revision, score_record.get("lineage") or {})
            conn.execute(
                """
                INSERT INTO candidate_scores (
                    run_id, model_id, revision, eligible, score,
                    blocking_reasons_json, score_breakdown_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    revision,
                    1 if score_record.get("eligible") else 0,
                    float(score_record.get("score") or 0.0),
                    _json(score_record.get("blocking_reasons") or []),
                    _json(score_record.get("score_breakdown") or {}),
                    now,
                ),
            )
            conn.commit()

    def start_run(
        self,
        run_id: str,
        model_id: str,
        revision: str | None = None,
        model_family: str | None = None,
        size_band: str | None = None,
        artifact_uri: str | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            self._upsert_model_revision(
                conn,
                model_id,
                revision,
                {"model_family": model_family, "size_band": size_band},
            )
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, model_id, revision, model_family, size_band,
                    started_at, status, artifact_uri
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, model_family, size_band, now, "started", artifact_uri),
            )
            conn.commit()

    def complete_run(
        self,
        run_id: str,
        status: str = "completed",
        selected_backend: str | None = None,
        selected_dtype: str | None = None,
        selected_quantization: str | None = None,
        base_score: float | None = None,
        adapted_score: float | None = None,
        adaptation_lift: float | None = None,
        latency_summary: str | None = None,
        context_drift_summary: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET completed_at = ?, status = ?,
                    selected_backend = COALESCE(?, selected_backend),
                    selected_dtype = COALESCE(?, selected_dtype),
                    selected_quantization = COALESCE(?, selected_quantization),
                    base_score = COALESCE(?, base_score),
                    adapted_score = COALESCE(?, adapted_score),
                    adaptation_lift = COALESCE(?, adaptation_lift),
                    latency_summary = COALESCE(?, latency_summary),
                    context_drift_summary = COALESCE(?, context_drift_summary)
                WHERE run_id = ?
                """,
                (
                    utc_now_iso(),
                    status,
                    selected_backend,
                    selected_dtype,
                    selected_quantization,
                    base_score,
                    adapted_score,
                    adaptation_lift,
                    latency_summary,
                    context_drift_summary,
                    run_id,
                ),
            )
            conn.commit()

    def record_latency_results(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        summary: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO latency_results (
                    run_id, model_id, revision, latency_summary, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, _json(summary), _json(records), utc_now_iso()),
            )
            conn.commit()

    def record_quality_results(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        summary: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO quality_results (
                    run_id, model_id, revision, base_score, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    revision,
                    summary.get("avg_quality_score"),
                    _json({"summary": summary, "records": records}),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def record_context_drift_results(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        summary: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_drift_results (
                    run_id, model_id, revision, context_drift_summary, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, _json(summary), _json(records), utc_now_iso()),
            )
            conn.commit()

    def record_finetune_run(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO finetune_runs (
                    run_id, model_id, revision, status, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, status, _json(payload), utc_now_iso()),
            )
            conn.commit()

    def record_adaptation_result(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        payload: dict[str, Any],
    ) -> None:
        overall = payload.get("overall") or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO adaptation_results (
                    run_id, model_id, revision, adapted_score, adaptation_lift,
                    payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    revision,
                    overall.get("adapted_quality_score"),
                    overall.get("adaptation_lift"),
                    _json(payload),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def record_metadata_result(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        claims_path: str | Path,
    ) -> None:
        payload = Path(claims_path).read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metadata_results (
                    run_id, model_id, revision, claims_vs_observed_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, payload, utc_now_iso()),
            )
            conn.commit()

    def record_runtime_result(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        decision: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_results (
                    run_id, model_id, revision, selected_backend, selected_dtype,
                    selected_quantization, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    revision,
                    decision.get("backend"),
                    decision.get("dtype"),
                    decision.get("quantization"),
                    _json(payload),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def record_failure(
        self,
        run_id: str | None,
        model_id: str,
        revision: str | None,
        failure_category: str,
        failure_message: str,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO failures (
                    run_id, model_id, revision, failure_category, failure_message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, failure_category, failure_message, now),
            )
            if run_id is not None:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, failure_category = ?, failure_message = ?, completed_at = ?
                    WHERE run_id = ?
                    """,
                    ("failed", failure_category, failure_message, now, run_id),
                )
            conn.commit()

    def record_artifact(
        self,
        run_id: str,
        artifact_type: str,
        artifact_uri: str,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    run_id, model_id, revision, artifact_type, artifact_uri, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, model_id, revision, artifact_type, artifact_uri, utc_now_iso()),
            )
            conn.commit()

    def record_comparison_result(
        self,
        run_id: str,
        model_id: str,
        revision: str | None,
        compared_to_model_id: str | None,
        compared_to_revision: str | None,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO comparisons (
                    run_id, model_id, revision, compared_to_model_id,
                    compared_to_revision, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    revision,
                    compared_to_model_id,
                    compared_to_revision,
                    _json(payload),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def fetch_leaderboard_rows(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    run_id, model_id, revision, model_family, size_band, status,
                    base_score, adapted_score, adaptation_lift, latency_summary,
                    context_drift_summary, artifact_uri, completed_at
                FROM runs
                WHERE status IN ('completed', 'failed')
                   OR base_score IS NOT NULL
                   OR adapted_score IS NOT NULL
                ORDER BY COALESCE(adapted_score, base_score, -1) DESC, completed_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_reporting_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            runs = conn.execute(
                """
                SELECT * FROM runs
                WHERE status IN ('completed', 'failed')
                   OR base_score IS NOT NULL
                   OR adapted_score IS NOT NULL
                ORDER BY COALESCE(adapted_score, base_score, -1) DESC, completed_at DESC
                """
            ).fetchall()
            return [self._enrich_run_for_reporting(conn, dict(row)) for row in runs]

    def fetch_reporting_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            return self._enrich_run_for_reporting(conn, dict(row))

    def fetch_tested_runs_for_model(self, model_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE model_id = ?
                  AND (status IN ('completed', 'failed')
                       OR base_score IS NOT NULL
                       OR adapted_score IS NOT NULL)
                ORDER BY completed_at DESC, started_at DESC
                """,
                (model_id,),
            ).fetchall()
            return [self._enrich_run_for_reporting(conn, dict(row)) for row in rows]

    def fetch_tested_comparison_pool(self, exclude_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE run_id != ?
                  AND (status = 'completed'
                       OR base_score IS NOT NULL
                       OR adapted_score IS NOT NULL)
                ORDER BY completed_at DESC, started_at DESC
                """,
                (exclude_run_id,),
            ).fetchall()
            return [self._enrich_run_for_reporting(conn, dict(row)) for row in rows]

    def already_tested_revision(self, model_id: str, revision: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM runs
                WHERE model_id = ? AND revision = ? AND status IN ('completed', 'failed')
                LIMIT 1
                """,
                (model_id, revision),
            ).fetchone()
        return row is not None

    def get_sync_fingerprint(self, run_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_fingerprint FROM result_sync_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return str(row["content_fingerprint"]) if row else None

    def replace_synced_run(
        self,
        payload: dict[str, Any],
        source_uri: str,
        content_fingerprint: str,
    ) -> None:
        run = payload["run"]
        run_id = str(run["run_id"])
        model_id = str(run["model_id"])
        revision = run.get("revision")
        now = utc_now_iso()
        with self._connect() as conn:
            self._upsert_model_revision(
                conn,
                model_id,
                revision,
                {
                    "model_family": run.get("model_family"),
                    "size_band": run.get("size_band"),
                },
            )
            for table in (
                "metadata_results",
                "runtime_results",
                "latency_results",
                "quality_results",
                "context_drift_results",
                "finetune_runs",
                "adaptation_results",
                "failures",
                "artifacts",
            ):
                conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, model_id, revision, model_family, size_band,
                    started_at, completed_at, status, failure_category, failure_message,
                    selected_backend, selected_dtype, selected_quantization,
                    base_score, adapted_score, adaptation_lift,
                    latency_summary, context_drift_summary, artifact_uri
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    revision,
                    run.get("model_family"),
                    run.get("size_band"),
                    run["started_at"],
                    run.get("completed_at"),
                    run["status"],
                    run.get("failure_category"),
                    run.get("failure_message"),
                    run.get("selected_backend"),
                    run.get("selected_dtype"),
                    run.get("selected_quantization"),
                    run.get("base_score"),
                    run.get("adapted_score"),
                    run.get("adaptation_lift"),
                    _json(payload.get("latency_summary") or {}),
                    _json(payload.get("context_summary") or {}),
                    source_uri,
                ),
            )
            self._insert_synced_payloads(conn, payload, now)
            conn.execute(
                """
                INSERT INTO result_sync_state (
                    run_id, source_uri, content_fingerprint, synced_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    source_uri = excluded.source_uri,
                    content_fingerprint = excluded.content_fingerprint,
                    synced_at = excluded.synced_at
                """,
                (run_id, source_uri, content_fingerprint, now),
            )
            conn.commit()

    def _insert_synced_payloads(
        self,
        conn: sqlite3.Connection,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        run = payload["run"]
        identity = (run["run_id"], run["model_id"], run.get("revision"))
        metadata = payload.get("metadata")
        if metadata is not None:
            conn.execute(
                """
                INSERT INTO metadata_results (
                    run_id, model_id, revision, claims_vs_observed_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (*identity, _json(metadata), created_at),
            )
        runtime = payload.get("runtime")
        if runtime is not None:
            decision = runtime.get("decision") or {}
            conn.execute(
                """
                INSERT INTO runtime_results (
                    run_id, model_id, revision, selected_backend, selected_dtype,
                    selected_quantization, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *identity,
                    decision.get("backend"),
                    decision.get("dtype"),
                    decision.get("quantization"),
                    _json(runtime),
                    created_at,
                ),
            )
        self._insert_synced_evaluations(conn, payload, identity, created_at)
        adaptation = payload.get("adaptation")
        if adaptation is not None:
            overall = adaptation.get("overall") or {}
            conn.execute(
                """
                INSERT INTO finetune_runs (
                    run_id, model_id, revision, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (*identity, adaptation.get("status", "unknown"), _json(adaptation), created_at),
            )
            conn.execute(
                """
                INSERT INTO adaptation_results (
                    run_id, model_id, revision, adapted_score, adaptation_lift,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *identity,
                    overall.get("adapted_quality_score"),
                    overall.get("adaptation_lift"),
                    _json(adaptation),
                    created_at,
                ),
            )
        failure = payload.get("failure")
        if failure:
            conn.execute(
                """
                INSERT INTO failures (
                    run_id, model_id, revision, failure_category, failure_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    *identity,
                    failure["failure_category"],
                    failure["failure_message"],
                    created_at,
                ),
            )
        for artifact_type, artifact_uri in payload.get("artifacts", {}).items():
            conn.execute(
                """
                INSERT INTO artifacts (
                    run_id, model_id, revision, artifact_type, artifact_uri, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (*identity, artifact_type, artifact_uri, created_at),
            )

    @staticmethod
    def _insert_synced_evaluations(
        conn: sqlite3.Connection,
        payload: dict[str, Any],
        identity: tuple[Any, Any, Any],
        created_at: str,
    ) -> None:
        latency_records = payload.get("latency_records")
        if latency_records is not None:
            conn.execute(
                """
                INSERT INTO latency_results (
                    run_id, model_id, revision, latency_summary, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    *identity,
                    _json(payload.get("latency_summary") or {}),
                    _json(latency_records),
                    created_at,
                ),
            )
        quality_records = payload.get("quality_records")
        if quality_records is not None:
            quality_summary = payload.get("quality_summary") or {}
            conn.execute(
                """
                INSERT INTO quality_results (
                    run_id, model_id, revision, base_score, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    *identity,
                    quality_summary.get("avg_quality_score"),
                    _json({"summary": quality_summary, "records": quality_records}),
                    created_at,
                ),
            )
        context_records = payload.get("context_records")
        if context_records is not None:
            conn.execute(
                """
                INSERT INTO context_drift_results (
                    run_id, model_id, revision, context_drift_summary,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    *identity,
                    _json(payload.get("context_summary") or {}),
                    _json(context_records),
                    created_at,
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _upsert_model_revision(
        self,
        conn: sqlite3.Connection,
        model_id: str,
        revision: str | None,
        payload: dict[str, Any],
    ) -> None:
        now = utc_now_iso()
        family = payload.get("model_family") or payload.get("family")
        size_band = payload.get("size_band")
        conn.execute(
            """
            INSERT INTO models (model_id, model_family, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                model_family = COALESCE(excluded.model_family, models.model_family)
            """,
            (model_id, family, now),
        )
        if revision:
            conn.execute(
                """
                INSERT INTO model_revisions (
                    model_id, revision, model_family, size_band, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_id, revision) DO UPDATE SET
                    model_family = COALESCE(excluded.model_family, model_revisions.model_family),
                    size_band = COALESCE(excluded.size_band, model_revisions.size_band)
                """,
                (model_id, revision, family, size_band, now),
            )

    def _enrich_run_for_reporting(
        self,
        conn: sqlite3.Connection,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        run_id = run["run_id"]
        model_id = run["model_id"]
        revision = run.get("revision")
        score = conn.execute(
            """
            SELECT eligible, score, blocking_reasons_json, score_breakdown_json
            FROM candidate_scores
            WHERE model_id = ? AND (revision = ? OR ? IS NULL)
            ORDER BY id DESC
            LIMIT 1
            """,
            (model_id, revision, revision),
        ).fetchone()
        artifacts = conn.execute(
            "SELECT artifact_type, artifact_uri FROM artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        latest_payloads: dict[str, sqlite3.Row | None] = {}
        for table in (
            "metadata_results",
            "runtime_results",
            "latency_results",
            "quality_results",
            "context_drift_results",
            "finetune_runs",
            "adaptation_results",
        ):
            latest_payloads[table] = conn.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        enriched = dict(run)
        enriched["artifacts"] = {row["artifact_type"]: row["artifact_uri"] for row in artifacts}
        if score is not None:
            enriched["scan_score"] = score["score"]
            enriched["eligible"] = bool(score["eligible"])
            enriched["blocking_reasons"] = _loads(score["blocking_reasons_json"], [])
            enriched["score_breakdown"] = _loads(score["score_breakdown_json"], {})
        else:
            enriched["scan_score"] = None
            enriched["eligible"] = None
            enriched["blocking_reasons"] = []
            enriched["score_breakdown"] = {}
        for key, row in latest_payloads.items():
            enriched[key] = dict(row) if row is not None else None
        return enriched


def initialize_db(db_path: str | Path) -> SQLiteStore:
    store = SQLiteStore(db_path)
    store.initialize()
    return store


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default
