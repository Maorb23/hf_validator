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
    _extract_job_id,
    build_worker_command,
    build_worker_env,
    manifest_uri_for_job,
    poll_nebius_job,
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
        "--format",
        "json",
        "--no-progress",
        "--args",
        "scripts/run_model_worker_job.sh",
    ]


def test_submit_nebius_job_forwards_worker_env_but_runs_cli_with_controller_env() -> None:
    payload = WorkerJobPayload(run_id="run_1", model_id="org/model", artifact_prefix="prefix")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, stdout='{"id": "aijob-e001"}', stderr=""
        )

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
    assert result.job_id == "aijob-e001"
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


def test_extract_job_id_uses_aijob_id_from_truncated_json() -> None:
    stdout = '{"id":"aijob-e00g5fr7eqxekzgsm0","nested":{"id":"job-6kk9vhi35y"'

    assert _extract_job_id(stdout) == "aijob-e00g5fr7eqxekzgsm0"


def test_extract_job_id_rejects_generic_nested_job_id() -> None:
    assert _extract_job_id('{"nested":{"id":"job-6kk9vhi35y"') is None


def test_poll_follows_logs_then_reads_final_status() -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "logs" in command:
            return subprocess.CompletedProcess(command, 0, stdout=None, stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"status": {"state": "SUCCEEDED"}}',
            stderr="",
        )

    status = poll_nebius_job(
        "job-1",
        NebiusJobConfig(dry_run=False),
        run_command=fake_run,
    )

    assert status == "completed"
    assert calls[0][0] == ["nebius", "ai", "job", "logs", "job-1", "--follow"]
    assert calls[1][0] == [
        "nebius",
        "ai",
        "job",
        "get",
        "job-1",
        "--format",
        "json",
    ]


def test_poll_does_not_fail_when_follow_succeeds_but_final_get_fails() -> None:
    def fake_run(command, **kwargs):
        if "logs" in command:
            return subprocess.CompletedProcess(command, 0, stdout=None, stderr="")
        raise subprocess.CalledProcessError(53, command, stderr="temporary lookup failure")

    status = poll_nebius_job(
        "job-1",
        NebiusJobConfig(dry_run=False),
        run_command=fake_run,
    )

    assert status == "completed_logs_followed"


def test_poll_reports_stderr_when_follow_and_lookup_fail() -> None:
    def fake_run(command, **kwargs):
        if "logs" in command:
            raise subprocess.CalledProcessError(1, command, stderr="log stream failed")
        raise subprocess.CalledProcessError(53, command, stderr="job lookup failed")

    with pytest.raises(NebiusJobError) as exc:
        poll_nebius_job(
            "job-1",
            NebiusJobConfig(dry_run=False),
            run_command=fake_run,
        )

    assert "log stream failed" in str(exc.value)
    assert "job lookup failed" in str(exc.value)
