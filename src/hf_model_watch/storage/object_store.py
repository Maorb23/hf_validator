from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.tracing import emit_trace_event


class ArtifactUploadError(RuntimeError):
    failure_category = FailureCategory.ARTIFACT_UPLOAD_FAILED.value


@dataclass(frozen=True)
class UploadResult:
    uri: str
    dry_run: bool
    uploaded_files: int
    skipped: bool = False


@dataclass(frozen=True)
class ObjectStoreConfig:
    bucket: str
    prefix: str
    endpoint_url: str | None = None
    dry_run: bool = True
    enabled: bool = True


def preflight_object_store_upload(run_id: str, config: ObjectStoreConfig) -> None:
    if not config.enabled:
        emit_trace_event(
            "artifact_upload_preflight",
            "artifact_upload",
            "skipped",
            "object storage upload disabled",
            metadata={"run_id": run_id},
        )
        return
    if config.dry_run:
        emit_trace_event(
            "artifact_upload_preflight",
            "artifact_upload",
            "skipped",
            "object storage dry-run enabled",
            metadata={"run_id": run_id},
        )
        return
    if not config.bucket:
        raise ArtifactUploadError("S3 bucket is required for artifact upload")
    missing = [key for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.getenv(key)]
    if missing:
        raise ArtifactUploadError(f"missing required S3 credential env vars: {', '.join(missing)}")
    try:
        import boto3
    except ImportError as exc:
        raise ArtifactUploadError(
            "Install boto3 to upload artifacts to S3-compatible storage"
        ) from exc
    try:
        client_kwargs = {"endpoint_url": config.endpoint_url} if config.endpoint_url else {}
        client = boto3.client("s3", **client_kwargs)
        key = _join_key(config.prefix, run_id, "_preflight.txt")
        client.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=b"hf-model-watch object-store preflight\n",
        )
    except Exception as exc:
        emit_trace_event(
            "artifact_upload_preflight",
            "artifact_upload",
            "failed",
            "object storage preflight failed",
            metadata={
                "run_id": run_id,
                "failure_category": ArtifactUploadError.failure_category,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise ArtifactUploadError(f"{type(exc).__name__}: {exc}") from exc
    emit_trace_event(
        "artifact_upload_preflight",
        "artifact_upload",
        "completed",
        "object storage preflight completed",
        metadata={"run_id": run_id},
        artifacts={"destination": f"s3://{config.bucket}/{key}"},
    )


def write_text_artifact(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def copy_local_artifact(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def artifact_uri(bucket: str, prefix: str, run_id: str) -> str:
    key = _join_key(prefix, run_id)
    return f"s3://{bucket}/{key}/"


def upload_run_artifacts(
    source_dir: str | Path,
    run_id: str,
    config: ObjectStoreConfig,
) -> UploadResult:
    source = Path(source_dir)
    if not config.enabled or not config.bucket:
        emit_trace_event(
            "artifact_upload",
            "artifact_upload",
            "skipped",
            "object storage upload disabled or bucket missing",
            metadata={"run_id": run_id},
        )
        return UploadResult(uri="", dry_run=config.dry_run, uploaded_files=0, skipped=True)
    if not source.exists() or not source.is_dir():
        raise ArtifactUploadError(f"artifact directory not found: {source}")

    destination_uri = artifact_uri(config.bucket, config.prefix, run_id)
    files = [path for path in source.rglob("*") if path.is_file()]
    emit_trace_event(
        "artifact_upload_started",
        "artifact_upload",
        "started",
        "uploading run artifacts",
        metadata={"run_id": run_id, "dry_run": config.dry_run},
        metrics={"files": len(files)},
        artifacts={"source": str(source), "destination": destination_uri},
    )
    if config.dry_run:
        emit_trace_event(
            "artifact_upload_completed",
            "artifact_upload",
            "completed",
            "dry-run artifact upload completed",
            metadata={"run_id": run_id},
            metrics={"files": len(files)},
            artifacts={"destination": destination_uri},
        )
        return UploadResult(uri=destination_uri, dry_run=True, uploaded_files=len(files))

    try:
        import boto3
    except ImportError as exc:
        raise ArtifactUploadError(
            "Install boto3 to upload artifacts to S3-compatible storage"
        ) from exc

    try:
        client_kwargs = {"endpoint_url": config.endpoint_url} if config.endpoint_url else {}
        client = boto3.client("s3", **client_kwargs)
        key_prefix = _join_key(config.prefix, run_id)
        for path in files:
            relative = path.relative_to(source).as_posix()
            client.upload_file(str(path), config.bucket, _join_key(key_prefix, relative))
    except Exception as exc:
        emit_trace_event(
            "artifact_upload_failed",
            "artifact_upload",
            "failed",
            "artifact upload failed",
            metadata={
                "run_id": run_id,
                "failure_category": ArtifactUploadError.failure_category,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            artifacts={"destination": destination_uri},
        )
        raise ArtifactUploadError(f"{type(exc).__name__}: {exc}") from exc

    emit_trace_event(
        "artifact_upload_completed",
        "artifact_upload",
        "completed",
        "artifact upload completed",
        metadata={"run_id": run_id},
        metrics={"files": len(files)},
        artifacts={"destination": destination_uri},
    )
    return UploadResult(uri=destination_uri, dry_run=False, uploaded_files=len(files))


def object_store_config_from_mapping(
    mapping: dict[str, Any],
    artifact_prefix: str = "",
    dry_run: bool | None = None,
) -> ObjectStoreConfig:
    s3 = mapping.get("s3") if isinstance(mapping.get("s3"), dict) else {}
    bucket = str(s3.get("bucket") or mapping.get("bucket") or "")
    prefix = str(artifact_prefix or s3.get("prefix") or mapping.get("prefix") or "hf-model-watch")
    endpoint = (
        s3.get("endpoint_url")
        or mapping.get("endpoint_url")
        or os.getenv("AWS_ENDPOINT_URL_S3")
        or os.getenv("AWS_ENDPOINT_URL")
    )
    enabled = bool(s3.get("enabled", mapping.get("enabled", bool(bucket))))
    dry = bool(s3.get("dry_run", mapping.get("dry_run", True))) if dry_run is None else dry_run
    return ObjectStoreConfig(
        bucket=bucket,
        prefix=prefix.strip("/"),
        endpoint_url=str(endpoint) if endpoint else None,
        dry_run=dry,
        enabled=enabled,
    )


def upload_s3_compatible(
    source: str | Path,
    bucket: str,
    key: str,
    endpoint_url: str | None = None,
    dry_run: bool = True,
) -> UploadResult:
    source_path = Path(source)
    prefix, _, run_id = key.strip("/").rpartition("/")
    if source_path.is_dir():
        return upload_run_artifacts(
            source_path,
            run_id or source_path.name,
            ObjectStoreConfig(bucket, prefix, endpoint_url, dry_run),
        )
    uri = f"s3://{bucket}/{key.strip('/')}"
    if dry_run:
        return UploadResult(uri=uri, dry_run=True, uploaded_files=1)
    try:
        import boto3
    except ImportError as exc:
        raise ArtifactUploadError(
            "Install boto3 to upload artifacts to S3-compatible storage"
        ) from exc
    client_kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    client = boto3.client("s3", **client_kwargs)
    try:
        client.upload_file(str(source_path), bucket, key)
    except Exception as exc:
        raise ArtifactUploadError(f"{type(exc).__name__}: {exc}") from exc
    return UploadResult(uri=uri, dry_run=False, uploaded_files=1)


def _join_key(*parts: str) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
