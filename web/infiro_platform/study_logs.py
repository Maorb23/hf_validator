"""Bounded, owner-scoped reads of saved worker logs. Never submit provider jobs."""

import re

from .models import Prepared
from .security import unseal
from .services import connections


def scrub_log(text, secrets):
    for value in secrets:
        if value:
            text = text.replace(value, "[redacted]")
    text = re.sub(r"https?://[^\s\"<>]+", lambda m: m[0].split("?", 1)[0], text)
    text = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[redacted]", text)
    text = re.sub(
        r"(?i)((?:WANDB_API_KEY|api_key|access_key|secret_key|token)\s*[=:]\s*)[^\s,\"']+",
        r"\1[redacted]",
        text,
    )
    # Strip terminal control sequences; React renders this as text as well.
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def log_reasons(text, expected):
    reasons = []
    for line in text.splitlines():
        match = re.search(r"Optuna trial (\d+) failed .*: (.+)$", line)
        if match:
            reasons.append(f"Trial {match[1]}: {match[2]}")
    successes = re.findall(r"Successful requests:\s+(\d+)", text)
    if successes and int(successes[-1]) < expected:
        reasons.append(
            f"{successes[-1]}/{expected} requests succeeded. The benchmark is incomplete "
            "and is excluded. This log tail may not contain the individual request error."
        )
    return list(dict.fromkeys(reasons))[-20:]


def saved_logs(job):
    import boto3
    from botocore.config import Config
    from llm_optimizer.remote.adapters import S3Artifacts

    prepared = Prepared.objects.get(pk=job.target_id, owner=job.owner)
    found = connections(job.owner, job.connection_refs, required=("storage",))
    storage = found["storage"]
    values = unseal(storage)
    # Known credentials are scrubbed without ever returning them to the browser.
    secrets = [v for c in found.values() for v in unseal(c).values() if isinstance(v, str)]
    client = boto3.client(
        "s3",
        aws_access_key_id=values["access_key_id"],
        aws_secret_access_key=values["secret_access_key"],
        aws_session_token=values.get("session_token"),
        region_name=storage.metadata["region"],
        endpoint_url=storage.metadata.get("endpoint") or None,
        config=Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 0}),
    )
    store = S3Artifacts(prepared.payload["bucket"], prepared.payload["prefix"], client=client)
    frameworks = []
    try:
        for framework in prepared.choices["frameworks"]:
            if framework not in {"vllm", "sglang", "trtllm"}:
                continue
            text = store.execution_log_tail(job.controller_run, framework)
            text = scrub_log((text or "")[-16384:], secrets)
            frameworks.append(
                {
                    "framework": framework,
                    "log_available": bool(text),
                    "log_tail": text,
                    "failure_reasons": log_reasons(text, prepared.choices["num_prompts"]),
                }
            )
    finally:
        client.close()
    return {"study_id": str(job.id), "frameworks": frameworks}
