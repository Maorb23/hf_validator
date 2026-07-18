from __future__ import annotations

import base64
import json
import subprocess

import pytest

from hf_model_watch.errors import FailureCategory
from hf_model_watch.orchestration.job_payloads import WorkerJobPayload
from hf_model_watch.orchestration.nebius_jobs import (
    WORKER_ENV_KEYS,
    NebiusJobConfig,
    NebiusJobError,
    _build_nebius_cli_command,
    build_worker_command,
    build_worker_env,
    manifest_uri_for_job,
    submit_nebius_job,
    validate_service_account_env,
)


def test_nebius_worker_payload_env_construction() -> None:
    payload = WorkerJobPayload(
        run_id="run_1",
        model_id="org/model",
        revision="abc",
        artifact_prefix="prefix",
    )
    env = build_worker_env(
        payload,
        NebiusJobConfig(dry_run=True),
        {
            "HF_TOKEN": "secret",
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret-key",
            "UNRELATED": "ignored",
        },
    )
    decoded = json.loads(base64.b64decode(env["PAYLOAD_JSON_B64"]).decode("utf-8"))

    assert env["RUN_ID"] == "run_1"
    assert env["MODEL_ID"] == "org/model"
    assert env["MODEL_REVISION"] == "abc"
    assert env["AWS_S3_PREFIX"] == "prefix"
    assert env["AWS_ACCESS_KEY_ID"] == "key"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret-key"
    assert env["HF_TOKEN"] == "secret"
    assert "UNRELATED" not in env
    assert decoded["model_id"] == "org/model"


def test_worker_env_whitelist_is_explicit() -> None:
    assert "AWS_ACCESS_KEY_ID" in WORKER_ENV_KEYS
    assert "AWS_SECRET_ACCESS_KEY" in WORKER_ENV_KEYS
    assert "AWS_ENDPOINT_URL_S3" in WORKER_ENV_KEYS
    assert "AWS_S3_BUCKET" in WORKER_ENV_KEYS


def test_build_worker_command_uses_worker_script() -> None:
    config = NebiusJobConfig(worker_script="scripts/run_model_worker_job.sh")

    assert build_worker_command(config) == [
        "bash",
        "scripts/run_model_worker_job.sh",
    ]


def test_build_nebius_cli_command_uses_ai_job_create() -> None:
    config = NebiusJobConfig(
        profile_name="profile",
        preset="1gpu-16vcpu-96gb",
        gpu_type="gpu-h100-sxm",
        worker_script="scripts/run_model_worker_job.sh",
    )

    assert _build_nebius_cli_command(["bash", "scripts/run_model_worker_job.sh"], config) == [
        "nebius",
        "ai",
        "job",
        "create",
        "--platform",
        "gpu-h100-sxm",
        "--preset",
        "1gpu-16vcpu-96gb",
        "--container-command",
        "bash",
        "--disk-size",
        "450Gi",
        "--timeout",
        "3h0m0s",
        "--args",
        "scripts/run_model_worker_job.sh",
    ]


def test_submit_nebius_job_forwards_worker_env_but_runs_cli_with_controller_env() -> None:
    payload = WorkerJobPayload(run_id="run_1", model_id="org/model", artifact_prefix="prefix")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout='{"id": "job-1"}', stderr="")

    controller_env = {
        "NB_PROJECT_ID": "project",
        "NEBIUS_CONFIG_YAML_B64": "config",
        "AWS_ACCESS_KEY_ID": "aws-key",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "AWS_ENDPOINT_URL_S3": "https://storage.eu-north1.nebius.cloud",
        "AWS_S3_BUCKET": "bucket",
        "UNRELATED": "controller-only",
    }

    result = submit_nebius_job(
        payload,
        NebiusJobConfig(dry_run=False),
        env=controller_env,
        run_command=fake_run,
    )

    command, kwargs = calls[0]
    assert result.job_id == "job-1"
    assert kwargs["env"] == controller_env
    assert "--env" in command
    assert "AWS_ACCESS_KEY_ID=aws-key" in command
    assert "AWS_SECRET_ACCESS_KEY=aws-secret" in command
    assert "UNRELATED=controller-only" not in command


def test_missing_service_account_vars_fail_typed_in_non_dry_run() -> None:
    with pytest.raises(NebiusJobError) as exc:
        validate_service_account_env({}, dry_run=False)

    assert exc.value.failure_category == FailureCategory.CONTROLLER_AUTH_FAILED.value


def test_dry_run_does_not_require_credentials() -> None:
    payload = WorkerJobPayload(run_id="run_1", model_id="org/model")
    result = submit_nebius_job(payload, NebiusJobConfig(dry_run=True), env={})

    assert result.status == "submitted_dry_run"
    assert result.job_id == "dry-run-run_1"
    assert result.dry_run is True


def test_manifest_uri_for_job() -> None:
    assert (
        manifest_uri_for_job("bucket", "prefix", "run_1")
        == "s3://bucket/prefix/run_1/run_manifest.json"
    )
