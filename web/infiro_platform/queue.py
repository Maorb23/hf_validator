"""One durable SQLite queue and one worker; every operation uses a fresh process."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta

from django.conf import settings
from django.db.models import F
from django.utils import timezone

from .diagnostics import has_validation_failure
from .limits import SHAREGPT_DISK_BYTES
from .models import Connection, Dataset, Job, Prepared, Recommendation
from .security import private_dir, unseal, workload_cache_dir
from .services import connections
from .workload_cache import CACHE_FULL, CACHE_INVALID, directory_bytes
from .workloads import FORMAT_ERROR, MEMORY_ERROR

TIME_LIMIT = 300
DISK_LIMIT = 256 * 1024 * 1024
ENV_FIELDS = {
    "nebius_inference": {"api_key": "NEBIUS_API_KEY"},
    "nebius_jobs": {"iam_token": "NEBIUS_IAM_TOKEN", "private_key": "NEBIUS_PRIVATE_KEY"},
    "huggingface": {"token": "HF_TOKEN"},
    "storage": {
        "access_key_id": "AWS_ACCESS_KEY_ID",
        "secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "session_token": "AWS_SESSION_TOKEN",
    },
    "tavily": {"api_key": "TAVILY_API_KEY"},
}
REQUIRED_KINDS = {
    "discover": {"nebius_inference", "huggingface", "tavily"},
    "dataset": {"huggingface"},
    "prepare": {"nebius_inference", "huggingface", "tavily", "nebius_jobs"},
    "study": {"nebius_jobs", "storage", "huggingface"},
    "validate": set(ENV_FIELDS),
}


def mark_cancelled(job):
    Job.objects.filter(pk=job.pk).update(state="cancelled", lease_until=None)
    model = {"discover": Recommendation, "dataset": Dataset, "prepare": Prepared}.get(job.kind)
    if model:
        model.objects.filter(pk=job.target_id, owner=job.owner).update(status="cancelled")


def child_environment(job):
    # Deliberate allowlist: no ambient HF/AWS/Nebius credentials, proxies, CLI
    # profiles, dotenv paths, Python startup hooks or server login home.
    env = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONUNBUFFERED": "1",
        "DJANGO_SETTINGS_MODULE": "infiro_site.settings",
        "INFIRO_DATABASE": str(settings.DATABASES["default"]["NAME"]),
        "INFIRO_PRIVATE_ROOT": str(settings.INFIRO_PRIVATE_ROOT),
        "INFIRO_PROVIDER_FACTORY": settings.INFIRO_PROVIDER_FACTORY,
        "INFIRO_SERVING_IMAGES": json.dumps(settings.INFIRO_SERVING_IMAGES),
        "INFIRO_IMAGE_ALIASES": json.dumps(settings.INFIRO_IMAGE_ALIASES),
        "INFIRO_WANDB": json.dumps(settings.INFIRO_WANDB),
        "INFIRO_LIVE_ENABLED": "1" if settings.INFIRO_LIVE_ENABLED else "0",
        "PYTHONPATH": os.pathsep.join(str(p) for p in sys.path if p),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0",
        "PYTHON_DOTENV_DISABLED": "1",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    env["INFIRO_WORKER_PID"] = str(os.getpid())
    root = private_dir(job.owner_id, job.id)
    for name, subdir in {
        "HOME": "home",
        "TMPDIR": "tmp",
        "HF_HOME": "hf",
        "HF_DATASETS_CACHE": "datasets",
        "XDG_CACHE_HOME": "cache",
        "AWS_CONFIG_FILE": "aws-config",
        "AWS_SHARED_CREDENTIALS_FILE": "aws-credentials",
    }.items():
        path = root / subdir
        if "FILE" not in name:
            path.mkdir(mode=0o700, exist_ok=True)
        env[name] = str(path)
    large_workload = job.kind == "dataset"
    if job.kind == "prepare":
        prepared = Prepared.objects.filter(pk=job.target_id, owner=job.owner).first()
        large_workload = bool(
            prepared
            and (prepared.dataset_id or prepared.choices.get("workload_mode") == "sharegpt")
        )
    if large_workload:
        env["INFIRO_LARGE_WORKLOAD"] = "1"
        env["INFIRO_SHAREGPT_CACHE"] = str(workload_cache_dir(job.owner_id))
    found = connections(job.owner, job.connection_refs)
    if {k: c.version for k, c in found.items()} != job.connection_versions:
        raise ValueError("Connection changed since this operation was queued.")
    for kind, connection in found.items():
        if kind == "nebius_jobs" and job.kind == "prepare":
            if Prepared.objects.get(pk=job.target_id, owner=job.owner).choices["dry_run"]:
                continue
        if kind not in REQUIRED_KINDS[job.kind]:
            continue
        for key, value in unseal(connection).items():
            env[ENV_FIELDS[kind][key]] = value
        if kind == "nebius_inference":
            env["NEBIUS_API_BASE_URL"] = connection.metadata["endpoint"]
        elif kind == "nebius_jobs":
            env["NEBIUS_PROJECT_ID"] = connection.metadata["project_id"]
        elif kind == "storage":
            env["AWS_DEFAULT_REGION"] = connection.metadata["region"]
            if connection.metadata.get("endpoint"):
                env["AWS_ENDPOINT_URL_S3"] = connection.metadata["endpoint"]
    return env


PERMANENT_INPUT_ERRORS = {
    FORMAT_ERROR, MEMORY_ERROR, CACHE_FULL, CACHE_INVALID,
    "Selected dataset files exceed the 1 GiB download limit, or sizes are unknown.",
    "Tokenizer exceeds the 128 MiB limit, or file sizes are unknown.",
    "Dataset exceeds the 10,000 row scan limit; choose a smaller split.",
    "Dataset row exceeds the 256 KiB limit.",
    "Dataset exceeds the 2,000,000 source token scan limit.",
    "Frozen workload exceeds the 16 MiB limit.",
    "Dataset has too few valid prompts; reduce prompt count or choose another split.",
    "Expected nonempty conversations [{from, value}]",
    "Invalid ShareGPT role or empty text",
    "ShareGPT conversation has no human prompt",
    "Custom HF dataset/tokenizer loading or access failed; check access and dependencies",
    "Custom HF dataset iteration or tokenization failed",
    "Selected JSON data file was not found at this dataset revision.",
    "Download limit exceeded (1 GiB per workload operation).",
}


def fail(job, message):
    job.refresh_from_db()
    print(
        json.dumps(
            {
                "job": str(job.id),
                "kind": job.kind,
                "attempt": job.attempts,
                "failure": job.progress.get("diagnostics", {}),
                "message": message,
            }
        ),
        flush=True,
    )
    invalid_configuration = has_validation_failure(job.progress.get("diagnostics", {}))
    if job.kind == "study" and job.controller_run:
        # Never forget a provider-side job after polling errors or credential
        # deletion. Keep it visible as attention_required for operator recovery.
        state = "attention_required" if invalid_configuration or job.attempts >= 3 else "queued"
    else:
        permanent = invalid_configuration or message in PERMANENT_INPUT_ERRORS
        state = "failed" if job.attempts >= 3 or permanent else "queued"
    Job.objects.filter(pk=job.pk).update(
        state=state,
        error=message,
        available_at=timezone.now() + timedelta(seconds=2**job.attempts),
        lease_until=None,
    )
    if state == "failed":
        model = {"discover": Recommendation, "dataset": Dataset, "prepare": Prepared}.get(job.kind)
        if model:
            model.objects.filter(pk=job.target_id, owner=job.owner).update(status="failed")
        if job.kind == "dataset":
            Dataset.objects.filter(pk=job.target_id, owner=job.owner).update(error=message)
        if job.kind == "validate":
            Connection.objects.filter(pk=job.target_id, owner=job.owner).update(validation="failed")


def recover(kinds=None):
    expired = Job.objects.filter(state="running", lease_until__lt=timezone.now())
    if kinds is not None:
        expired = expired.filter(kind__in=kinds)
    for job in expired:
        fail(job, "The previous worker stopped. The operation will recover from persisted state.")


def claim(kinds=None):
    now = timezone.now()
    eligible = Job.objects.filter(state="queued", available_at__lte=now)
    if kinds is not None:
        eligible = eligible.filter(kind__in=kinds)
    candidate = (
        eligible
        .order_by("available_at", "created_at")
        .first()
    )
    if not candidate:
        return None
    changed = Job.objects.filter(pk=candidate.pk, state="queued").update(
        state="running",
        attempts=F("attempts") + 1,
        lease_until=now + timedelta(seconds=TIME_LIMIT + 30),
        error="",
    )
    if not changed:
        return None
    candidate.refresh_from_db()
    return candidate


def run_once(*, kinds=None):
    recover(kinds)
    job = claim(kinds)
    if not job:
        return False
    if job.cancel_requested and job.kind != "study":
        mark_cancelled(job)
        return True
    try:
        env = child_environment(job)
        previous_provider_diagnostics = job.progress.get("provider_diagnostics")
        process = subprocess.Popen(
            [sys.executable, "-m", "infiro_platform.child", str(job.id)],
            env=env,
            cwd=settings.INFIRO_PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        started = time.monotonic()
        root = private_dir(job.owner_id, job.id)
        disk_limit = (
            SHAREGPT_DISK_BYTES
            if env.get("INFIRO_LARGE_WORKLOAD") == "1"
            else DISK_LIMIT
        )
        while process.poll() is None:
            job.refresh_from_db(fields=["cancel_requested"])
            size = directory_bytes(root)
            if env.get("INFIRO_SHAREGPT_CACHE"):
                size += directory_bytes(env["INFIRO_SHAREGPT_CACHE"])
            cancelled = job.cancel_requested and job.kind != "study"
            if cancelled or time.monotonic() - started > TIME_LIMIT or size > disk_limit:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if cancelled:
                    mark_cancelled(job)
                    return True
                limit = "1.25 GiB" if disk_limit == SHAREGPT_DISK_BYTES else "256 MiB"
                raise ValueError(f"Process exceeded its execution time or {limit} disk limit.")
            time.sleep(0.2)
        job.refresh_from_db()
        if (
            job.progress.get("provider_diagnostics")
            and job.progress["provider_diagnostics"] != previous_provider_diagnostics
        ):
            print(
                json.dumps(
                    {
                        "job": str(job.id),
                        "kind": job.kind,
                        "provider_error": job.progress.get("provider_error"),
                        "provider_diagnostics": job.progress["provider_diagnostics"],
                    }
                ),
                flush=True,
            )
        if process.returncode != 0 or job.state == "running":
            fail(
                job,
                job.error
                or "Operation failed or exceeded a resource limit; check connections, "
                "dataset schema and configured limits.",
            )
    except Exception as exc:
        message = (
            "Operation could not start. Check connection versions, encryption key "
            "and worker configuration."
        )
        if (
            isinstance(exc, ValueError)
            and str(exc).startswith("Process exceeded its execution time or ")
        ):
            message = str(exc)
        fail(
            job,
            message,
        )
    return True
