"""Python boundary to the existing recommender, preparation and optimizer code."""

import json
import re

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import Artifact, Connection, Dataset, Job, Prepared, Recommendation
from .security import digest, private_dir

TERMINAL = {"completed", "failed", "cancelled", "dry_run"}


def owned(model, user, identifier):
    from django.http import Http404

    try:
        filters = {"pk": identifier, "owner": user}
        if any(f.name == "deleted_at" for f in model._meta.fields):
            filters["deleted_at__isnull"] = True
        return model.objects.get(**filters)
    except (model.DoesNotExist, ValueError, TypeError):
        raise Http404 from None


def connections(user, refs, *, required=()):
    if not isinstance(refs, dict) or set(refs) - {
        "nebius_inference",
        "nebius_jobs",
        "huggingface",
        "storage",
        "tavily",
    }:
        raise ValueError("Invalid connection references.")
    found = {}
    for kind, identifier in refs.items():
        connection = owned(Connection, user, identifier)
        if connection.kind != kind:
            raise ValueError("Connection type does not match its purpose.")
        found[kind] = connection
    if set(required) - set(found):
        raise ValueError("Required connections: " + ", ".join(required))
    return found


def enqueue(user, kind, target, refs, key):
    if target.owner_id != user.id:
        raise ValueError("Operation target belongs to a different workspace.")
    found = connections(user, refs)
    existing = Job.objects.filter(owner=user, dedupe_key=key).first()
    if existing:
        return existing
    if Job.objects.filter(owner=user).exclude(state__in=TERMINAL).count() >= 10:
        raise ValueError(
            "This workspace already has ten active operations. Finish or cancel one first."
        )
    job, _ = Job.objects.get_or_create(
        owner=user,
        dedupe_key=key,
        defaults={
            "kind": kind,
            "target_id": target.id,
            "connection_refs": refs,
            "connection_versions": {k: c.version for k, c in found.items()},
            "available_at": timezone.now(),
        },
    )
    return job


def binding(prepared):
    return {
        "payload": prepared.payload,
        "intent": prepared.intent,
        "choices": prepared.choices,
        "summary": prepared.summary,
        "connection_refs": prepared.connection_refs,
        "connection_versions": prepared.connection_versions,
        "recommendation_version": prepared.recommendation_version,
        "recommendation": str(prepared.recommendation_id),
        "dataset": str(prepared.dataset_id) if prepared.dataset_id else None,
        "owner": prepared.owner_id,
    }


def assert_current(prepared):
    rec = prepared.recommendation
    if (
        prepared.invalidated
        or rec.version != prepared.recommendation_version
        or prepared.digest != digest(binding(prepared))
    ):
        raise ValueError(
            "This approval is stale. Prepare and review the current configuration again."
        )
    found = connections(prepared.owner, prepared.connection_refs)
    if {k: c.version for k, c in found.items()} != prepared.connection_versions:
        raise ValueError("A connection changed. Prepare and review again.")
    return found


@transaction.atomic
def approve(user, identifier, reviewed_digest, explicit):
    prepared = owned(Prepared, user, identifier)
    if (
        explicit is not True
        or not prepared.digest
        or reviewed_digest != prepared.digest
        or prepared.status != "prepared"
    ):
        raise ValueError("Explicit approval of this exact prepared configuration is required.")
    if prepared.choices.get("workload_mode") != "sharegpt":
        raise ValueError("Random workloads were removed. Prepare a dataset-backed configuration.")
    assert_current(prepared)
    if not prepared.choices["dry_run"] and not (
        settings.INFIRO_LIVE_ENABLED and settings.INFIRO_PROVIDER_FACTORY
    ):
        raise ValueError(
            "Live submission is disabled: configure secure credential delivery "
            "and verified worker images first."
        )
    # Compare-and-set serializes concurrent approvals on SQLite. The unique queue
    # key is a second guard, including after a worker or HTTP process restarts.
    Prepared.objects.filter(pk=prepared.pk, invalidated=False).update(approved_at=timezone.now())
    return enqueue(
        user,
        "study",
        prepared,
        prepared.connection_refs,
        "approve:" + str(prepared.id) + ":" + prepared.digest,
    )


def redact(value):
    """Defence in depth for upstream reports, not a substitute for secret isolation."""
    import os

    secrets = [
        os.environ[k]
        for k in (
            "NEBIUS_API_KEY",
            "NEBIUS_IAM_TOKEN",
            "NEBIUS_PRIVATE_KEY",
            "HF_TOKEN",
            "TAVILY_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "WANDB_API_KEY",
        )
        if os.getenv(k)
    ]
    if isinstance(value, dict):
        return {
            k: redact(v)
            for k, v in value.items()
            if not any(
                word in k.lower()
                for word in ("api_key", "secret", "access_key", "iam_token", "signed_url")
            )
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
    return value


def write_artifact(job, name, value):
    root = private_dir(job.owner_id, job.id)
    if name == "workload.json" and redact(value) != value:
        raise ValueError(
            "Workload content contains a configured credential. Remove secrets from the dataset."
        )
    raw = json.dumps(redact(value), indent=2, allow_nan=False).encode()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Artifact exceeds 16 MiB.")
    import hashlib

    path = root / name
    path.write_bytes(raw)
    path.chmod(0o600)
    artifact, _ = Artifact.objects.update_or_create(
        owner=job.owner,
        job=job,
        name=name,
        defaults={
            "relative_path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        },
    )
    return artifact


def read_workload(dataset):
    artifact = Artifact.objects.get(
        owner=dataset.owner,
        job__owner=dataset.owner,
        job__kind="dataset",
        job__target_id=dataset.id,
        name="workload.json",
    )
    path = private_dir(dataset.owner_id, artifact.job_id) / artifact.relative_path
    raw = path.read_bytes()
    import hashlib

    if hashlib.sha256(raw).hexdigest() != artifact.sha256:
        raise ValueError("Stored workload failed its integrity check.")
    from llm_optimizer.hf_workload import validate_manifest

    return validate_manifest(json.loads(raw))


def perform_discovery(job):
    from hf_model_watch.recommender.core import DiscoveryRequestSpec, load_settings
    from hf_model_watch.recommender.discovery import discover_models

    rec = Recommendation.objects.get(pk=job.target_id, owner=job.owner)
    if rec.status in {"recommended", "empty"}:
        job.state = "completed"
        job.result = {"recommendation_id": str(rec.id), "status": rec.status}
        return
    directory = private_dir(job.owner_id, job.id)
    config = load_settings(settings.INFIRO_PROJECT_ROOT / "config/recommender.yaml")
    config["tavily"]["cache_dir"] = str(directory / "evidence_cache")
    report = discover_models(
        DiscoveryRequestSpec(**rec.requirements), config, directory / "discovery"
    )
    rec.report = redact(report)
    rec.status = "recommended" if report.get("recommendations") else "empty"
    rec.save(update_fields=["report", "status", "updated_at"])
    job.result = {"recommendation_id": str(rec.id), "status": rec.status}
    job.state = "completed"


def perform_model_selection(job):
    import os

    from huggingface_hub import HfApi

    rec = Recommendation.objects.get(pk=job.target_id, owner=job.owner)
    if not rec.selected:
        requested = rec.report["requested_model"]
        try:
            info = HfApi(token=os.environ.get("HF_TOKEN") or False).model_info(
                requested["model_id"], revision=requested["revision"])
        except Exception as exc:
            raise ValueError("Model access or revision lookup failed. Check the model ID, "
                             "revision and Hugging Face connection.") from exc
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", info.sha or ""):
            raise ValueError("The model lookup did not return a pinned commit.")
        rec.selected = {"model_id": requested["model_id"], "revision": info.sha}
        rec.status = "ready"
        rec.save(update_fields=["selected", "status", "updated_at"])
    job.state = "completed"
    job.result = {"recommendation_id": str(rec.id), "selected": rec.selected}


def perform_dataset(job):
    from .workloads import identity, materialize

    dataset = Dataset.objects.get(pk=job.target_id, owner=job.owner)
    if dataset.status == "ready":
        read_workload(dataset)  # Integrity-check the immutable saved sample.
        job.state = "completed"
        job.result = {"dataset_id": str(dataset.id), "identity": dataset.identity}
        return
    def progress(message):
        job.progress["stage"] = message
        job.save(update_fields=["progress"])

    dataset.status = "validating"
    dataset.save(update_fields=["status"])
    manifest = materialize(dataset.config, progress=progress)
    write_artifact(job, "workload.json", manifest)
    dataset.identity = identity(manifest)
    dataset.preview = [
        {"prompt": r["prompt"][:400], "prompt_len": r["prompt_len"], "output_len": r["output_len"]}
        for r in manifest["requests"][:3]
    ]
    dataset.status = "ready"
    dataset.error = ""
    dataset.save()
    job.state = "completed"
    job.result = {"dataset_id": str(dataset.id), "identity": dataset.identity}


def finalize_preparation(job, prepared, found):
    """A saved draft becomes approvable only after reference verification succeeds."""
    if prepared.summary.get("draft_sha256") != digest(
        {
            "payload": prepared.payload,
            "intent": prepared.intent,
            "workload_identity": prepared.summary["workload_identity"],
        }
    ):
        raise ValueError("Preparation draft failed its integrity check.")
    if {k: c.version for k, c in found.items()} != prepared.connection_versions:
        raise ValueError("A connection changed. Prepare and review again.")
    provider_settings = {k: c.metadata for k, c in found.items()}
    if not prepared.choices["dry_run"]:
        from .nebius_provider import pin_secret_references, validate_live_configuration

        job.progress["stage"] = "Resolving and verifying SecretStash references"
        job.save(update_fields=["progress"])
        validate_live_configuration(prepared, found, prepared.payload["images"])
        secret_connections = dict(found)
        tracking = prepared.summary.get("tracking", {})
        if tracking:
            from types import SimpleNamespace

            if not tracking.get("project") or not tracking.get("secret_version_id"):
                raise ValueError("W&B requires a project and SecretStash reference.")
            secret_connections["wandb"] = SimpleNamespace(metadata=tracking)
        provider_settings = pin_secret_references(secret_connections)
        if tracking:
            prepared.summary["tracking"] = provider_settings["wandb"]
    prepared.summary["provider_settings"] = {
        k: metadata
        for k, metadata in provider_settings.items()
        if k in {"nebius_jobs", "storage", "huggingface"}
    }
    prepared.summary["provider_verification"] = (
        "verified_references" if not prepared.choices["dry_run"] else "not_required_dry_run"
    )
    expected_trials = prepared.choices["trials"]
    if prepared.payload.get("optuna", {}).get("trials") != expected_trials or any(
        v.get("trials", expected_trials) != expected_trials
        for v in prepared.payload.get("optuna_by_framework", {}).values()
    ):
        raise ValueError("Prepared trial count differs from Configure. Prepare again.")
    optuna = prepared.payload.get("optuna", {})
    for key, default in (("sampler", "TPESampler"), ("pruner", "NopPruner")):
        if optuna.get(key, default) != prepared.choices.get(key, default) or any(
            override.get(key, optuna.get(key, default)) != prepared.choices.get(key, default)
            for override in prepared.payload.get("optuna_by_framework", {}).values()
        ):
            raise ValueError(f"Prepared {key} differs from Configure. Prepare again.")
    if prepared.choices.get("objectives") and (
        optuna.get("objectives") != prepared.choices["objectives"]
        or prepared.intent["context"]["objectives"] != prepared.choices["objectives"]
    ):
        raise ValueError("Prepared objectives differ from Configure. Prepare again.")
    expected_multivariate = prepared.choices.get("tpe_multivariate", False)
    kwargs = optuna.get("sampler_kwargs", {})
    if kwargs.get("multivariate", False) != expected_multivariate or any(
        override.get("sampler_kwargs", kwargs).get("multivariate", False) != expected_multivariate
        for override in prepared.payload.get("optuna_by_framework", {}).values()
    ):
        raise ValueError("Prepared multivariate TPE differs from Configure. Prepare again.")
    prepared.digest = digest(binding(prepared))
    prepared.status = "prepared"
    prepared.save()
    job.state = "completed"
    job.result = {"prepared_id": str(prepared.id)}


def perform_prepare(job):
    from hf_model_watch.optimization.preparation import prepare

    from .workloads import DEFAULT_DATASET, identity, materialize

    prepared = Prepared.objects.select_related("recommendation", "dataset").get(
        pk=job.target_id, owner=job.owner
    )
    if prepared.digest:
        # Recovery after a crash between saving the review and completing its
        # queue entry must never regenerate an already approvable request.
        if prepared.digest != digest(binding(prepared)):
            raise ValueError("Prepared configuration failed its integrity check.")
        job.state = "completed"
        job.result = {"prepared_id": str(prepared.id)}
        return
    if prepared.invalidated or prepared.recommendation.version != prepared.recommendation_version:
        raise ValueError("Selection changed. Prepare a new configuration.")
    if prepared.choices.get("workload_mode") != "sharegpt":
        raise ValueError("Random workloads are no longer supported. Select and prepare a dataset.")
    selected = prepared.recommendation.selected
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", selected.get("revision", "")):
        raise ValueError("The selected model needs a pinned commit revision.")
    found = connections(job.owner, prepared.connection_refs)
    if prepared.payload and prepared.summary.get("draft_sha256"):
        finalize_preparation(job, prepared, found)
        return

    def stage(name):
        job.progress["stage"] = name
        job.save(update_fields=["progress"])

    stage("Preparing frozen workload and tokenizer")
    if prepared.dataset_id:
        if prepared.dataset.owner_id != job.owner_id or prepared.dataset.status != "ready":
            raise ValueError("The selected private dataset is not ready.")
        manifest = read_workload(prepared.dataset)
        expected = prepared.dataset.config
        for key in ("num_prompts", "seed", "input_tokens", "output_tokens", "concurrency"):
            if expected[key] != prepared.choices[key]:
                raise ValueError("Workload settings changed; validate a new dataset sample first.")
        if (
            manifest["tokenizer"] != selected["model_id"]
            or manifest["tokenizer_revision"] != selected["revision"]
        ):
            raise ValueError(
                "Dataset tokenizer differs from the selected pinned model. Validate a new sample."
            )
    else:
        config = {
            **prepared.choices,
            "model": selected["model_id"],
            "model_revision": selected["revision"],
            "repo_id": DEFAULT_DATASET,
            "revision": "main",
            "config": None,
            "data_file": "ShareGPT_V3_unfiltered_cleaned_split.json",
            "split": "train",
        }
        manifest = materialize(config, progress=stage)
    if redact(manifest) != manifest:
        raise ValueError(
            "Workload content contains a configured credential. Remove secrets from the dataset."
        )
    values = {
        k: v
        for k, v in prepared.choices.items()
        if k not in {"spending_limit_usd", "timeout_minutes", "workload_mode"}
    }
    # Adapter provenance belongs to the immutable manifest, not HFDataset intake.
    values["dataset"] = {
        k: v for k, v in manifest["dataset"].items()
        if k not in {"source_format", "adapter_version"}
    }
    directory = private_dir(job.owner_id, job.id)
    storage = found.get("storage")
    execution = {
        "dry_run": prepared.choices["dry_run"],
        "bucket": storage.metadata["bucket"] if storage else "dry-run-no-storage",
        "prefix": f"infiro/users/{job.owner_id}/prepared/{prepared.id}",
        "timeout_minutes": prepared.choices["timeout_minutes"],
        "workload_manifest": manifest,
        "images": settings.INFIRO_SERVING_IMAGES,
        "dspark_compatibility": settings.INFIRO_DSPARK_COMPATIBILITY,
        "optuna": {"trials": prepared.choices["trials"],
                   "seed": prepared.choices["seed"],
                   "sampler": prepared.choices.get("sampler", "TPESampler"),
                   **({"sampler_kwargs": {"multivariate": True}}
                      if prepared.choices.get("tpe_multivariate") else {}),
                   "pruner": prepared.choices.get("pruner", "NopPruner")},
    }
    run = {
        "run_id": "run_" + prepared.recommendation.id.hex,
        "selected_model_id": selected["model_id"],
        "selection": selected,
        "request_spec": values,
    }
    stage("Researching model and runtime compatibility")
    result = prepare(
        run, {}, execution, directory, settings.INFIRO_PROJECT_ROOT / "config/recommender.yaml"
    )
    if result["status"] != "prepared":
        raise ValueError("Preparation requires more input or documented runtime evidence.")
    prepared.payload = json.loads((directory / "request.json").read_text())
    prepared.intent = json.loads((directory / "intent.json").read_text())
    prepared.summary = result["summary"] | {
        "tracking": dict(settings.INFIRO_WANDB) if not prepared.choices["dry_run"] else {},
        "workload_identity": identity(manifest),
        "workload_preview": [
            {
                "prompt": r["prompt"][:400],
                "prompt_len": r["prompt_len"],
                "output_len": r["output_len"],
            }
            for r in manifest["requests"][:3]
        ],
        "connections": {k: c.name for k, c in found.items()},
        "provider_settings": {
            k: c.metadata
            for k, c in found.items()
            if k in {"nebius_jobs", "storage", "huggingface"}
        },
        "spending_limit_usd": prepared.choices["spending_limit_usd"],
        "cost_label": "Estimated GPU allocation only; not a hard provider spending cap. "
        "Storage, CPU, network and pricing changes are excluded.",
        "research": json.loads((directory / "research.json").read_text()).get("inspection", {}),
        "data_origin": prepared.recommendation.report.get("data_origin", "provider_research"),
    }
    prepared.summary["provider_verification"] = "pending"
    prepared.summary["draft_sha256"] = digest(
        {
            "payload": prepared.payload,
            "intent": prepared.intent,
            "workload_identity": prepared.summary["workload_identity"],
        }
    )
    # These are private draft artifacts, not approval or submission authorization.
    write_artifact(job, "request.json", prepared.payload)
    write_artifact(job, "intent.json", prepared.intent)
    write_artifact(job, "workload.json", manifest)
    prepared.status = "draft"
    prepared.save()
    finalize_preparation(job, prepared, found)


def perform_study(job):
    from hf_model_watch.optimization.analysis import analyze
    from hf_model_watch.optimization.contracts import NewStudyIntent
    from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter

    from .controller import make_controller

    if job.cancel_requested and not job.controller_run:
        job.state = "cancelled"
        job.result = {
            "measurement_status": "no_measurements",
            "reason": "Cancelled before submission.",
        }
        return
    prepared = Prepared.objects.select_related("recommendation").get(
        pk=job.target_id, owner=job.owner
    )
    if not prepared.approved_at:
        raise ValueError("Run has no explicit approval.")
    if not job.controller_run:
        assert_current(prepared)
    controller = make_controller(job, prepared)
    if not job.controller_run:
        response = controller.submit(
            prepared.payload, idempotency_key=str(job.id) + ":" + prepared.digest
        )
        job.controller_run = response["run_id"]
        job.save(update_fields=["controller_run"])
    if job.cancel_requested:
        controller.cancel(job.controller_run)
    controller.tick()
    status = controller.status(job.controller_run)
    from hf_model_watch.optimization.timing import elapsed

    observations = job.progress.get("queue_observations", {})
    for framework_job in status["jobs"]:
        framework = framework_job["framework"]
        if framework_job["state"] == "running" and framework not in observations:
            observed_at = timezone.now().isoformat()
            observations[framework] = {"running_observed_at": observed_at,
                "queue_wait_s": elapsed(status.get("created_at"), observed_at)}
    job.progress = {
        "queue_observations": observations,
        "state": status["state"],
        **{
            key: job.progress[key]
            for key in ("provider_diagnostics", "provider_error",
                        "recovered_provider_diagnostics", "recovered_provider_error",
                        "provider_error_scopes")
            if key in job.progress
        },
        "jobs": [
            {k: j.get(k) for k in ("framework", "state", "progress", "error")}
            for j in status["jobs"]
        ],
    }
    # Keep the first running observation even if result retrieval must retry.
    job.save(update_fields=["progress"])
    intent = NewStudyIntent.model_validate(prepared.intent)
    summary = OptimizationAdapter(controller, {}).results(job.controller_run, intent.context)
    for framework in prepared.choices["frameworks"]:
        timing = summary.framework_timings.setdefault(framework, {
            "startup_to_ready_s": None, "benchmark_s": None})
        timing["queue_wait_s"] = observations.get(framework, {}).get("queue_wait_s")
    result = analyze(summary, intent.selection_policy)
    job.result = redact(
        {
            "optimization_settings": {
                "objectives": intent.context.objectives,
                "sampler": prepared.payload.get("optuna", {}).get("sampler", "TPESampler"),
                "sampler_kwargs": prepared.payload.get("optuna", {}).get("sampler_kwargs", {}),
                "pruner": prepared.payload.get("optuna", {}).get("pruner", "NopPruner"),
                "trials": prepared.payload.get("optuna", {}).get("trials"),
                "seed": intent.context.workload.seed,
            },
            "summary": summary.model_dump(mode="json"),
            "analysis": result.model_dump(mode="json"),
            "measurement_status": "unmeasured_dry_run"
            if prepared.choices["dry_run"]
            else "measured"
            if summary.candidates
            else "no_measurements",
        }
    )
    job.state = status["state"] if status["state"] in TERMINAL else "queued"
    if job.state in {"completed", "dry_run"}:
        from .study_status import display_progress

        job.progress = display_progress(job)
        job.error = ""
    if job.state in TERMINAL:
        write_artifact(job, "results.json", job.result)


def perform_validation(job):
    import os

    from .queue import ENV_FIELDS
    from .security import endpoint

    connection = Connection.objects.get(pk=job.target_id, owner=job.owner)
    values = {
        key: os.environ[env_key]
        for key, env_key in ENV_FIELDS[connection.kind].items()
        if os.getenv(env_key)
    }
    meta = connection.metadata
    import requests

    session = requests.Session()
    session.trust_env = False
    if connection.kind == "huggingface":
        from huggingface_hub import HfApi

        HfApi(token=values["token"]).whoami()
    elif connection.kind == "nebius_inference":
        response = session.get(
            endpoint(meta["endpoint"]) + "/models",
            headers={"Authorization": "Bearer " + values["api_key"]},
            timeout=15,
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise ValueError("Provider authentication validation failed.")
    elif connection.kind == "tavily":
        response = session.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": "Bearer " + values["api_key"]},
            timeout=15,
            allow_redirects=False,
        )
        response.raise_for_status()
    elif connection.kind == "storage":
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            aws_access_key_id=values["access_key_id"],
            aws_secret_access_key=values["secret_access_key"],
            aws_session_token=values.get("session_token"),
            region_name=meta["region"],
            endpoint_url=endpoint(meta["endpoint"]) if meta.get("endpoint") else None,
            config=Config(
                connect_timeout=10,
                read_timeout=10,
                retries={"max_attempts": 0},
                s3={"addressing_style": "path"},
            ),
        )
        client.head_bucket(Bucket=meta["bucket"])
    else:
        # IAM validation uses the operator's safe Python provider, never a CLI
        # profile that could inherit a different user's credentials.
        if not settings.INFIRO_PROVIDER_FACTORY:
            raise ValueError(
                "Configure the credential-safe Nebius provider adapter "
                "to validate project permissions."
            )
        import importlib

        module, name = settings.INFIRO_PROVIDER_FACTORY.rsplit(":", 1)
        getattr(importlib.import_module(module), name).validate_connection(connection, values)
    Connection.objects.filter(pk=connection.pk, version=connection.version).update(
        validation="valid", validated_at=timezone.now()
    )
    job.state = "completed"
    job.result = {
        "validation": "valid",
        "scope": "Authentication/read access only; job, object-write and GPU permissions "
        "are not established.",
    }


def perform(job):
    {
        "model": perform_model_selection,
        "discover": perform_discovery,
        "dataset": perform_dataset,
        "prepare": perform_prepare,
        "study": perform_study,
        "validate": perform_validation,
    }[job.kind](job)
