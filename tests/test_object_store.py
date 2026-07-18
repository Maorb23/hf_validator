from __future__ import annotations

import pytest

from hf_model_watch.errors import FailureCategory
from hf_model_watch.storage.object_store import (
    ArtifactUploadError,
    ObjectStoreConfig,
    artifact_uri,
    object_store_config_from_mapping,
    preflight_object_store_upload,
    upload_run_artifacts,
)


def test_object_store_dry_run_uploads_directory(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}", encoding="utf-8")

    result = upload_run_artifacts(
        run_dir,
        "run_1",
        ObjectStoreConfig(bucket="bucket", prefix="prefix", dry_run=True),
    )

    assert result.dry_run is True
    assert result.uploaded_files == 1
    assert result.uri == "s3://bucket/prefix/run_1/"


def test_artifact_uri_generation() -> None:
    assert artifact_uri("bucket", "prefix/nested", "run_1") == "s3://bucket/prefix/nested/run_1/"


def test_object_store_disabled_without_bucket_is_skipped(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()

    result = upload_run_artifacts(run_dir, "run_1", ObjectStoreConfig(bucket="", prefix="x"))

    assert result.skipped is True
    assert result.uri == ""


def test_upload_failure_maps_to_artifact_upload_failed(tmp_path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(ArtifactUploadError) as exc:
        upload_run_artifacts(
            missing_dir,
            "run_1",
            ObjectStoreConfig(bucket="bucket", prefix="prefix", dry_run=False),
        )

    assert exc.value.failure_category == FailureCategory.ARTIFACT_UPLOAD_FAILED.value


def test_object_store_config_from_mapping_prefers_payload_prefix() -> None:
    config = object_store_config_from_mapping(
        {"s3": {"enabled": True, "bucket": "bucket", "prefix": "old", "endpoint_url": "http://s3"}},
        artifact_prefix="new",
        dry_run=True,
    )

    assert config.bucket == "bucket"
    assert config.prefix == "new"
    assert config.endpoint_url == "http://s3"
    assert config.dry_run is True


def test_object_store_config_uses_endpoint_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://storage.eu-north1.nebius.cloud")

    config = object_store_config_from_mapping(
        {"s3": {"enabled": True, "bucket": "bucket", "prefix": "prefix"}},
        dry_run=False,
    )

    assert config.endpoint_url == "https://storage.eu-north1.nebius.cloud"


def test_preflight_upload_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    with pytest.raises(ArtifactUploadError, match="AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"):
        preflight_object_store_upload(
            "run_1",
            ObjectStoreConfig(bucket="bucket", prefix="prefix", dry_run=False),
        )


def test_preflight_upload_skips_dry_run(monkeypatch) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    preflight_object_store_upload(
        "run_1",
        ObjectStoreConfig(bucket="bucket", prefix="prefix", dry_run=True),
    )
