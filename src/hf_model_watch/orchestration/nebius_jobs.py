from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.orchestration.job_payloads import WorkerJobPayload

WORKER_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL_S3",
    "AWS_ENDPOINT_URL",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_S3_BUCKET",
    "HF_MODEL_WATCH_OBJECT_STORE_DRY_RUN",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
)


class NebiusJobError(RuntimeError):
    def __init__(self, message: str, failure_category: str) -> None:
        super().__init__(message)
        self.failure_category = failure_category


@dataclass(frozen=True)
class NebiusJobConfig:
    project_id: str | None = None
    profile_name: str = "hf-model-watch-controller"
    image: str | None = None
    preset: str | None = "1gpu-16vcpu-96gb"
    gpu_type: str = "gpu-h100-sxm"
    gpu_count: int = 1
    timeout_minutes: int = 180
    dry_run: bool = True
    worker_script: str = "scripts/run_model_worker_job.sh"
    artifact_prefix: str = "hf-model-watch"


@dataclass(frozen=True)
class NebiusJobResult:
    job_id: str
    status: str
    payload: dict[str, Any]
    command: list[str]
    env: dict[str, str]
    dry_run: bool
    failure_category: str | None = None


def build_worker_command(config: NebiusJobConfig) -> list[str]:
    return ["bash", config.worker_script]


def build_worker_env(
    payload: WorkerJobPayload,
    config: NebiusJobConfig,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    source_env = base_env or {}
    env = {key: source_env[key] for key in WORKER_ENV_KEYS if source_env.get(key)}
    payload_json = json.dumps(payload.model_dump(), sort_keys=True, separators=(",", ":"))
    env.update(
        {
            "RUN_ID": payload.run_id,
            "MODEL_ID": payload.model_id,
            "MODEL_REVISION": payload.revision or "",
            "PAYLOAD_JSON_B64": base64.b64encode(payload_json.encode("utf-8")).decode("ascii"),
            "AWS_S3_PREFIX": payload.artifact_prefix or config.artifact_prefix,
            "HF_MODEL_WATCH_OBJECT_STORE_DRY_RUN": str(config.dry_run).lower(),
        }
    )
    return env


def validate_service_account_env(env: dict[str, str], dry_run: bool) -> None:
    if dry_run:
        return
    has_config = bool(env.get("NEBIUS_CONFIG_YAML_B64"))
    has_key = bool(env.get("NEBIUS_SERVICE_ACCOUNT_KEY_B64"))
    has_keypair = bool(
        (env.get("SA_ID") or env.get("NEBIUS_SERVICE_ACCOUNT_ID"))
        and (env.get("PUBLIC_KEY_ID") or env.get("NEBIUS_PUBLIC_KEY_ID"))
        and (env.get("NEBIUS_PRIVATE_KEY_B64") or env.get("NEBIUS_PRIVATE_KEY_PATH"))
    )
    if not env.get("NB_PROJECT_ID"):
        raise NebiusJobError(
            "NB_PROJECT_ID is required for non-dry-run Nebius submission",
            FailureCategory.CONTROLLER_AUTH_FAILED.value,
        )
    if not (has_config or has_key or has_keypair):
        raise NebiusJobError(
            "Nebius service-account credentials are required for non-dry-run submission",
            FailureCategory.CONTROLLER_AUTH_FAILED.value,
        )


def submit_nebius_job(
    payload: WorkerJobPayload,
    config: NebiusJobConfig,
    env: dict[str, str] | None = None,
    run_command: Any = subprocess.run,
) -> NebiusJobResult:
    base_env = dict(os.environ if env is None else env)
    validate_service_account_env(base_env, config.dry_run)
    command = build_worker_command(config)
    worker_env = build_worker_env(payload, config, base_env)
    if config.dry_run:
        return NebiusJobResult(
            job_id=f"dry-run-{payload.run_id}",
            status="submitted_dry_run",
            payload=payload.model_dump(),
            command=command,
            env=_public_env(worker_env),
            dry_run=True,
        )

    nebius_command = _build_nebius_cli_command(command, config, worker_env)
    try:
        completed = run_command(
            nebius_command,
            check=True,
            capture_output=True,
            text=True,
            env=base_env,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise NebiusJobError(
            f"Nebius job submission failed: {detail}",
            FailureCategory.NEBIUS_JOB_FAILED.value,
        ) from exc
    except Exception as exc:
        raise NebiusJobError(
            f"Nebius job submission failed: {type(exc).__name__}: {exc}",
            FailureCategory.NEBIUS_JOB_FAILED.value,
        ) from exc
    return NebiusJobResult(
        job_id=_extract_job_id(completed.stdout) or f"submitted-{payload.run_id}",
        status="submitted",
        payload=payload.model_dump(),
        command=nebius_command,
        env=_public_env(worker_env),
        dry_run=False,
    )


def poll_nebius_job(
    job_id: str,
    config: NebiusJobConfig,
    env: dict[str, str] | None = None,
    run_command: Any = subprocess.run,
) -> str:
    if config.dry_run or job_id.startswith("dry-run-"):
        return "completed_dry_run"
    try:
        completed = run_command(
            ["nebius", "ai", "job", "get", "--id", job_id, "--format", "json"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except Exception as exc:
        raise NebiusJobError(
            f"Nebius job polling failed: {type(exc).__name__}: {exc}",
            FailureCategory.NEBIUS_JOB_FAILED.value,
        ) from exc
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return "unknown"
    return str(payload.get("status") or payload.get("state") or "unknown")


def _build_nebius_cli_command(
    worker_command: list[str],
    config: NebiusJobConfig,
    worker_env: dict[str, str] | None = None,
) -> list[str]:
    executable = worker_command[0]
    args = worker_command[1:]

    # Convert timeout_minutes to CLI duration format (e.g. "3h0m0s")
    hours, remainder = divmod(config.timeout_minutes, 60)
    timeout_str = f"{hours}h{remainder}m0s"

    command = [
        "nebius",
        "ai",
        "job",
        "create",
        "--platform", config.gpu_type,
        "--preset", config.preset or "1gpu-16vcpu-96gb",
        "--container-command", executable,
        "--disk-size", "450Gi",
        "--timeout", timeout_str,
    ]

    if config.project_id:
        command.extend(["--parent-id", config.project_id])
    if config.image:
        command.extend(["--image", config.image])
    for arg in args:
        command.extend(["--args", arg])

    # Pass env vars into the container via --env KEY=VALUE (repeatable)
    for key, value in (worker_env or {}).items():
        command.extend(["--env", f"{key}={value}"])

    return command


def _extract_job_id(stdout: str) -> str | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        value = payload.get("id") or payload.get("job_id")
        if value:
            return str(value)
    for token in stdout.split():
        if token.startswith("job-"):
            return token
    return None


def _public_env(env: dict[str, str]) -> dict[str, str]:
    redacted = {}
    secret_markers = ("TOKEN", "KEY", "SECRET", "B64")
    for key, value in env.items():
        redacted[key] = "***" if any(marker in key for marker in secret_markers) else value
    return redacted


def build_payload_for_candidate(
    candidate: dict[str, Any],
    run_id: str,
    config: dict[str, Any],
    artifact_prefix: str,
) -> WorkerJobPayload:
    return WorkerJobPayload(
        run_id=run_id,
        model_id=str(candidate["model_id"]),
        revision=candidate.get("revision"),
        config=config,
        artifact_prefix=artifact_prefix,
        skip_finetune=bool(config.get("skip_finetune", False)),
        finetune_smoke_only=bool(config.get("finetune_smoke_only", False)),
    )


def manifest_uri_for_job(bucket: str, prefix: str, run_id: str) -> str:
    clean_prefix = "/".join(part.strip("/") for part in (prefix, run_id) if part.strip("/"))
    return f"s3://{bucket}/{clean_prefix}/run_manifest.json"


def local_manifest_path(artifact_dir: str | Path, run_id: str) -> Path:
    return Path(artifact_dir) / run_id / "run_manifest.json"
