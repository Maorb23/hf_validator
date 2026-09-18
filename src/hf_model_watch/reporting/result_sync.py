from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hf_model_watch.evals.metrics import summarize_records
from hf_model_watch.evals.quality_eval import summarize_quality_records
from hf_model_watch.reporting.leaderboard import generate_leaderboard_markdown
from hf_model_watch.storage.schemas import utc_now_iso
from hf_model_watch.storage.sqlite_store import SQLiteStore
from hf_model_watch.tracing import emit_trace_event

REPORTING_FILES = {
    "run_manifest.json",
    "runtime_result.json",
    "adaptation_results.json",
    "adapter_eval.json",
    "latency_results.jsonl",
    "quality_results.jsonl",
    "context_drift_results.jsonl",
    "metadata/claims_vs_observed.json",
    "worker_failure.json",
}


@dataclass(frozen=True)
class SyncSummary:
    discovered_runs: int = 0
    imported_runs: int = 0
    skipped_runs: int = 0
    downloaded_files: int = 0
    failed_runs: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def ingest_local_results(
    input_dir: str | Path,
    db_path: str | Path,
    leaderboard_path: str | Path = "reports/leaderboard.md",
    force: bool = False,
) -> SyncSummary:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"results input directory does not exist: {root}")
    run_dirs = _discover_run_dirs(root)
    store = SQLiteStore(db_path)
    store.initialize()
    imported = skipped = failed = 0
    for run_dir in run_dirs:
        try:
            fingerprint = _local_fingerprint(run_dir)
            run_id = _manifest_identity(run_dir)["run_id"]
            if not force and store.get_sync_fingerprint(run_id) == fingerprint:
                skipped += 1
                continue
            payload = parse_run_artifacts(run_dir, source_uri=str(run_dir.resolve()))
            store.replace_synced_run(payload, str(run_dir.resolve()), fingerprint)
            imported += 1
        except Exception as exc:
            failed += 1
            emit_trace_event(
                "result_sync_run_failed",
                "result_sync",
                "failed",
                f"failed to import local run: {type(exc).__name__}: {exc}",
                metadata={"run_dir": str(run_dir)},
            )
    generate_leaderboard_markdown(db_path, leaderboard_path)
    summary = SyncSummary(len(run_dirs), imported, skipped, 0, failed)
    _emit_sync_completed(summary, str(root))
    return summary


def sync_s3_results(
    bucket: str,
    prefix: str,
    endpoint_url: str | None,
    db_path: str | Path,
    cache_dir: str | Path = "outputs/synced-nebius",
    leaderboard_path: str | Path = "reports/leaderboard.md",
    force: bool = False,
    s3_client: Any | None = None,
) -> SyncSummary:
    if not bucket:
        raise ValueError("S3 bucket is required for result synchronization")
    client = s3_client or _s3_client(endpoint_url)
    normalized_prefix = prefix.strip("/")
    objects = _list_s3_objects(client, bucket, normalized_prefix)
    runs = _group_s3_runs(objects, normalized_prefix)
    store = SQLiteStore(db_path)
    store.initialize()
    cache_root = Path(cache_dir)
    imported = skipped = downloaded = failed = 0
    for run_id, run_objects in sorted(runs.items()):
        fingerprint = _remote_fingerprint(run_objects)
        if not force and store.get_sync_fingerprint(run_id) == fingerprint:
            skipped += 1
            continue
        run_dir = cache_root / run_id
        try:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            key_root = _join_key(normalized_prefix, run_id)
            for item in run_objects:
                key = str(item["Key"])
                relative = key.removeprefix(f"{key_root}/")
                if relative not in REPORTING_FILES:
                    continue
                destination = run_dir / Path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(destination))
                downloaded += 1
            source_uri = f"s3://{bucket}/{key_root}/"
            payload = parse_run_artifacts(run_dir, source_uri=source_uri)
            store.replace_synced_run(payload, source_uri, fingerprint)
            imported += 1
        except Exception as exc:
            failed += 1
            emit_trace_event(
                "result_sync_run_failed",
                "result_sync",
                "failed",
                f"failed to synchronize S3 run: {type(exc).__name__}: {exc}",
                metadata={"run_id": run_id, "bucket": bucket},
            )
    generate_leaderboard_markdown(db_path, leaderboard_path)
    summary = SyncSummary(len(runs), imported, skipped, downloaded, failed)
    _emit_sync_completed(summary, f"s3://{bucket}/{normalized_prefix}/")
    return summary


def parse_run_artifacts(run_dir: str | Path, source_uri: str) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _read_json(root / "run_manifest.json", required=True)
    runtime = _read_json(root / "runtime_result.json")
    adaptation = _read_json(root / "adaptation_results.json")
    metadata = _read_json(root / "metadata" / "claims_vs_observed.json")
    worker_failure = _read_json(root / "worker_failure.json")
    latency_records = _read_jsonl(root / "latency_results.jsonl")
    quality_records = _read_jsonl(root / "quality_results.jsonl")
    context_records = _read_jsonl(root / "context_drift_results.jsonl")
    run_id = str(manifest["run_id"])
    model_id = str(manifest["model_id"])
    revision = manifest.get("revision") or (runtime or {}).get("revision")
    runtime_metadata = (runtime or {}).get("metadata") or {}
    decision = (runtime or {}).get("decision") or {}
    family = _model_family(runtime_metadata)
    size_band = _size_band(runtime_metadata.get("estimated_params_billion"))
    quality_summary = _quality_summary(quality_records)
    latency_summary = _latency_summary(latency_records)
    context_summary = _context_summary(context_records)
    overall = (adaptation or {}).get("overall") or {}
    failure = _failure_payload(manifest, adaptation, worker_failure)
    status = _run_status(manifest, adaptation, failure)
    completed_at = (
        (adaptation or {}).get("created_at")
        or manifest.get("completed_at")
        or utc_now_iso()
    )
    return {
        "run": {
            "run_id": run_id,
            "model_id": model_id,
            "revision": revision,
            "model_family": family,
            "size_band": size_band,
            "started_at": manifest.get("started_at") or completed_at,
            "completed_at": completed_at,
            "status": status,
            "failure_category": (failure or {}).get("failure_category"),
            "failure_message": (failure or {}).get("failure_message"),
            "selected_backend": decision.get("backend"),
            "selected_dtype": decision.get("dtype"),
            "selected_quantization": decision.get("quantization"),
            "base_score": (
                quality_summary.get("math_accuracy")
                if quality_summary.get("math_accuracy") is not None
                else quality_summary.get("sentence_embedding_similarity_mean")
            ),
            "adapted_score": overall.get("adapted_quality_score"),
            "adaptation_lift": overall.get("adaptation_lift"),
        },
        "metadata": metadata,
        "runtime": runtime,
        "latency_records": latency_records,
        "latency_summary": latency_summary,
        "quality_records": quality_records,
        "quality_summary": quality_summary,
        "context_records": context_records,
        "context_summary": context_summary,
        "adaptation": adaptation,
        "failure": failure,
        "artifacts": _artifact_uris(manifest, run_id, source_uri),
    }


def _discover_run_dirs(root: Path) -> list[Path]:
    if (root / "run_manifest.json").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("run_manifest.json")})


def _manifest_identity(run_dir: Path) -> dict[str, str]:
    manifest = _read_json(run_dir / "run_manifest.json", required=True)
    return {"run_id": str(manifest["run_id"]), "model_id": str(manifest["model_id"])}


def _local_fingerprint(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for relative in sorted(REPORTING_FILES):
        path = run_dir / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_json(path: Path, required: bool = False) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required run artifact is missing: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record {line_number} must be an object: {path}")
        records.append(record)
    return records


def _quality_summary(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    if records is None:
        return {}
    return summarize_quality_records(records)


def _latency_summary(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    if records is None:
        return {}
    summary = summarize_records(records)
    summary["avg_decode_tokens_per_second"] = _average(records, "decode_tokens_per_second")
    summary["avg_ttft_ms"] = _average(records, "ttft_ms")
    return summary


def _context_summary(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    if records is None:
        return {}
    summary = summarize_records(records)
    success_rate = sum(1 for record in records if record.get("success")) / max(len(records), 1)
    summary.update(
        {
            "success_rate": success_rate,
            "drift_score": 1.0 - success_rate,
            "claimed_context": max(
                (int(record.get("claimed_context") or 0) for record in records),
                default=0,
            ),
            "actual_context_tested": max(
                (int(record.get("actual_context_tested") or 0) for record in records),
                default=0,
            ),
        }
    )
    return summary


def _average(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def _model_family(metadata: dict[str, Any]) -> str | None:
    model_type = (metadata.get("config") or {}).get("model_type")
    if model_type:
        return str(model_type)
    architecture = metadata.get("architecture") or []
    return str(architecture[0]) if architecture else None


def _size_band(value: Any) -> str | None:
    if value is None:
        return None
    size = float(value)
    if size <= 3:
        return "<=3B"
    if size <= 7:
        return "3B-7B"
    if size <= 15:
        return "7B-15B"
    return "15B-30B"


def _failure_payload(
    manifest: dict[str, Any],
    adaptation: dict[str, Any] | None,
    worker_failure: dict[str, Any] | None,
) -> dict[str, str] | None:
    if worker_failure:
        return {
            "failure_category": str(worker_failure.get("failure_category") or "unknown_failure"),
            "failure_message": str(worker_failure.get("failure_message") or "worker failed"),
        }
    manifest_failure = manifest.get("failure")
    if isinstance(manifest_failure, dict):
        return {
            "failure_category": str(manifest_failure.get("failure_category") or "unknown_failure"),
            "failure_message": str(manifest_failure.get("failure_message") or "run failed"),
        }
    if adaptation and not adaptation.get("training_completed", False):
        status = str(adaptation.get("status") or "")
        if status not in {"skipped", "smoke_completed"}:
            smoke = adaptation.get("smoke") or {}
            return {
                "failure_category": str(smoke.get("failure_category") or "finetune_failed"),
                "failure_message": str(smoke.get("message") or status or "fine-tune failed"),
            }
    return None


def _run_status(
    manifest: dict[str, Any],
    adaptation: dict[str, Any] | None,
    failure: dict[str, str] | None,
) -> str:
    if failure:
        return "failed"
    if adaptation and (
        adaptation.get("training_completed")
        or adaptation.get("status") in {"skipped", "smoke_completed"}
    ):
        return "completed"
    return str(manifest.get("status") or "completed")


def _artifact_uris(
    manifest: dict[str, Any],
    run_id: str,
    source_uri: str,
) -> dict[str, str]:
    artifacts = {"run_root": source_uri}
    for artifact_type, original in (manifest.get("artifacts") or {}).items():
        normalized = str(original).replace("\\", "/")
        marker = f"/{run_id}/"
        relative = normalized.split(marker, maxsplit=1)[-1] if marker in normalized else normalized
        if relative.startswith("artifacts/"):
            relative = relative.split(f"{run_id}/", maxsplit=1)[-1]
        artifacts[str(artifact_type)] = f"{source_uri.rstrip('/')}/{relative.lstrip('/')}"
    return artifacts


def _s3_client(endpoint_url: str | None) -> Any:
    import boto3

    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    return boto3.client("s3", **kwargs)


def _list_s3_objects(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"):
        objects.extend(page.get("Contents") or [])
    return objects


def _group_s3_runs(
    objects: list[dict[str, Any]],
    prefix: str,
) -> dict[str, list[dict[str, Any]]]:
    root = f"{prefix.rstrip('/')}/" if prefix else ""
    grouped: dict[str, list[dict[str, Any]]] = {}
    manifest_runs = {
        str(item["Key"])[len(root) :].split("/", maxsplit=1)[0]
        for item in objects
        if str(item["Key"]).startswith(root)
        and str(item["Key"]).endswith("/run_manifest.json")
    }
    for item in objects:
        relative = str(item["Key"])[len(root) :]
        run_id = relative.split("/", maxsplit=1)[0]
        if run_id in manifest_runs:
            grouped.setdefault(run_id, []).append(item)
    return grouped


def _remote_fingerprint(objects: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(objects, key=lambda value: str(value["Key"])):
        key = str(item["Key"])
        if not any(key.endswith(f"/{name}") for name in REPORTING_FILES):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(str(item.get("ETag") or "").encode("utf-8"))
        digest.update(str(item.get("Size") or 0).encode("ascii"))
    return digest.hexdigest()


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _emit_sync_completed(summary: SyncSummary, source: str) -> None:
    emit_trace_event(
        "result_sync_completed",
        "result_sync",
        "completed" if summary.failed_runs == 0 else "failed",
        "result synchronization completed",
        metadata={"source": source},
        metrics=summary.as_dict(),
    )
