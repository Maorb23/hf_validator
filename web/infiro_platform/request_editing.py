"""Validated revisions of unapproved requests using the optimizer's contracts."""

import copy
import json

from django.db import transaction
from llm_optimizer.remote.contracts import OptimizationRequest

from .contracts import Choices
from .models import Prepared
from .security import digest
from .services import assert_current, enqueue, owned, redact, write_artifact

EDITABLE = {
    "gpu_type",
    "gpu_count",
    "preset",
    "timeout_minutes",
    "server_args",
    "optuna",
    "optuna_by_framework",
    "semantic_context",
}
SAMPLER_KWARGS = {
    "multivariate",
    "group",
    "constant_liar",
    "n_startup_trials",
    "n_ei_candidates",
    "population_size",
    "deterministic_objective",
}
PRUNER_KWARGS = {
    "n_startup_trials",
    "n_warmup_steps",
    "interval_steps",
    "n_min_trials",
    "min_resource",
    "max_resource",
    "reduction_factor",
    "min_early_stopping_rate",
    "bootstrap_count",
}


def validate_search(settings, choices, objectives):
    # No arbitrary component imports, RDB URLs, workload overrides or constructors.
    allowed = {
        "trials",
        "sampler",
        "sampler_kwargs",
        "pruner",
        "pruner_kwargs",
        "objectives",
        "seed",
        "min_completed_requests",
        "timeout",
    }
    if set(settings) - allowed:
        raise ValueError("Unsupported search setting in request editor.")
    if settings.get("objectives", objectives) != objectives:
        raise ValueError("Framework overrides cannot change objectives.")
    if settings.get("seed", choices["seed"]) != choices["seed"]:
        raise ValueError("Keep the frozen workload seed for the sampler.")
    if "timeout" in settings and (
        type(settings["timeout"]) not in (int, float) or settings["timeout"] <= 0
    ):
        raise ValueError("Search timeout must be a positive number of seconds.")
    if settings.get("sampler_kwargs", {}).get("multivariate", False) != choices["tpe_multivariate"]:
        raise ValueError("Framework overrides must retain the reviewed multivariate TPE setting.")
    for key in ("trials", "sampler", "pruner"):
        if settings.get(key, choices[key]) != choices[key]:
            raise ValueError(
                "Framework overrides must retain the reviewed trials, sampler and pruner."
            )
    for key, allowed_keys in (("sampler_kwargs", SAMPLER_KWARGS), ("pruner_kwargs", PRUNER_KWARGS)):
        kwargs = settings.get(key, {})
        if not isinstance(kwargs, dict) or set(kwargs) - allowed_keys:
            raise ValueError(f"Unsupported {key}; use native numeric or boolean options.")
        for name, value in kwargs.items():
            if name in {"multivariate", "group", "constant_liar", "deterministic_objective"}:
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a boolean.")
            elif type(value) is not int or value < (
                2
                if name in {"population_size", "reduction_factor"}
                else 1
                if name in {"interval_steps", "n_min_trials", "n_ei_candidates"}
                else 0
            ):
                raise ValueError(f"{name} must be a supported nonnegative integer.")
        if key == "sampler_kwargs":
            compatible = {
                "TPESampler": {
                    "multivariate",
                    "group",
                    "constant_liar",
                    "n_startup_trials",
                    "n_ei_candidates",
                },
                "GPSampler": {"n_startup_trials", "deterministic_objective"},
                "NSGAIISampler": {"population_size"},
                "RandomSampler": set(),
                "GridSampler": set(),
            }
            if set(kwargs) - compatible[choices["sampler"]]:
                raise ValueError("Sampler options are incompatible with the chosen sampler.")
            if kwargs.get("group") and not kwargs.get("multivariate"):
                raise ValueError("TPE group sampling requires multivariate=true.")
        else:
            compatible = {
                "NopPruner": set(),
                "MedianPruner": {
                    "n_startup_trials",
                    "n_warmup_steps",
                    "interval_steps",
                    "n_min_trials",
                },
                "SuccessiveHalvingPruner": {
                    "min_resource",
                    "reduction_factor",
                    "min_early_stopping_rate",
                    "bootstrap_count",
                },
                "HyperbandPruner": {
                    "min_resource",
                    "max_resource",
                    "reduction_factor",
                    "bootstrap_count",
                },
            }
            if set(kwargs) - compatible[choices["pruner"]]:
                raise ValueError("Pruner options are incompatible with the chosen pruner.")
            if any(kwargs.get(k, 1) < 1 for k in ("min_resource", "max_resource")):
                raise ValueError("Pruning resources must be positive.")
            if kwargs.get("max_resource", kwargs.get("min_resource", 1)) < kwargs.get(
                "min_resource", 1
            ):
                raise ValueError("max_resource must be at least min_resource.")


@transaction.atomic
def revise_request(user, identifier, reviewed_digest, raw):
    source = owned(Prepared, user, identifier)
    if (
        source.status != "prepared"
        or source.approved_at
        or not reviewed_digest
        or reviewed_digest != source.digest
    ):
        raise ValueError("Only the current unapproved request can be edited. Reload its review.")
    assert_current(source)
    if not isinstance(raw, dict):
        raise ValueError("Send request.json as a JSON object.")
    try:
        if len(json.dumps(raw, allow_nan=False).encode()) > 16 * 1024**2:
            raise ValueError("Request exceeds 16 MiB.")
    except (TypeError, OverflowError) as exc:
        raise ValueError("Request must contain JSON values.") from exc
    if redact(raw) != raw:
        raise ValueError("Remove credential values from request.json.")
    changed = {
        key for key in raw.keys() | source.payload.keys() if raw.get(key) != source.payload.get(key)
    }
    if changed - EDITABLE:
        raise ValueError(
            "Protected fields changed: "
            + ", ".join(sorted(changed - EDITABLE))
            + ". Use Configure or select another model to change these."
        )
    if not changed:
        raise ValueError("The request has no changes.")
    payload = OptimizationRequest.parse(copy.deepcopy(raw)).to_dict()
    optuna = payload["optuna"]
    values = {
        **source.choices,
        **{key: payload[key] for key in ("gpu_type", "gpu_count", "timeout_minutes")},
        **{
            key: optuna.get(key, default)
            for key, default in (
                ("trials", 30),
                ("sampler", "TPESampler"),
                ("pruner", "NopPruner"),
                ("objectives", []),
            )
        },
        "tpe_multivariate": optuna.get("sampler_kwargs", {}).get("multivariate", False),
    }
    choices = Choices.model_validate(values).model_dump()
    for key in ("objectives", "trials", "sampler", "pruner"):
        optuna[key] = choices[key]
    if optuna.get("seed", choices["seed"]) != choices["seed"]:
        raise ValueError("Keep the frozen workload seed for the sampler.")
    for settings in [
        optuna,
        *({**optuna, **overrides} for overrides in payload["optuna_by_framework"].values()),
    ]:
        validate_search(settings, choices, choices["objectives"])
    if any(key not in payload["frameworks"] for key in payload["optuna_by_framework"]):
        raise ValueError("Search overrides must use selected frameworks.")
    from llm_optimizer.args import ArgScope, parse_args_str
    from llm_optimizer.optimization import OptunaSearchSpace
    from llm_optimizer.predefined import SERVER_CONFIGS

    for framework, args in payload["server_args"].items():
        prior = source.payload.get("server_args", {}).get(framework, [])
        for arg in args:
            if arg in prior:
                continue
            if "range(" in arg:
                raise ValueError("Use bounded explicit lists for serving argument values.")
            parsed = parse_args_str(arg, ArgScope.SERVER, SERVER_CONFIGS[framework], False)
            names = {
                name
                for item in parsed
                for name in (item.name if isinstance(item.name, tuple) else (item.name,))
            }
            if names & {"model", "model_path", "revision", "huggingface_model_revision"} or any(
                "speculative" in name or "draft" in name for name in names
            ):
                raise ValueError(
                    "Model identity and speculative settings require a new preparation."
                )
        parsed = parse_args_str(";".join(args), ArgScope.SERVER, SERVER_CONFIGS[framework], False)
        OptunaSearchSpace(tuple(parsed))  # Reject duplicate/overlapping dimensions natively.
        if choices["speculative_decoding"] == "require" and any(
            arg not in args for arg in prior if "speculative" in arg or "draft" in arg
        ):
            raise ValueError("Required speculative settings cannot be removed.")
    intent = copy.deepcopy(source.intent)
    intent["context"].update(
        gpu_type=choices["gpu_type"],
        gpu_count=choices["gpu_count"],
        objectives=choices["objectives"],
    )
    from hf_model_watch.optimization.contracts import NewStudyIntent

    intent = NewStudyIntent.model_validate(intent).model_dump(mode="json")
    if raw.get("semantic_context") not in (
        source.payload.get("semantic_context"),
        intent["context"],
    ):
        raise ValueError(
            "Semantic context must retain the pinned model and workload "
            "and match the edited settings."
        )
    payload["semantic_context"] = intent["context"]
    hourly = choices["gpu_count"] * choices["gpu_hourly_cost_usd"] * len(choices["frameworks"])
    summary = copy.deepcopy(source.summary)
    if "server_args" in changed:
        summary.update(
            serving_settings=[],
            serving_settings_status="user_edited_not_verified",
            search_profile="user_edited",
        )
    summary.update(
        {
            key: payload[key]
            for key in (
                "gpu_type",
                "gpu_count",
                "preset",
                "timeout_minutes",
                "optuna",
                "optuna_by_framework",
            )
        }
    )
    summary.update(
        maximum_hourly_gpu_cost_usd=hourly,
        maximum_gpu_cost_usd=hourly * choices["timeout_minutes"] / 60,
        request_edits=sorted(changed),
        provider_verification="pending",
        server_search_dimensions={
            f: [a.partition("=")[0] for a in args] for f, args in payload["server_args"].items()
        },
    )
    revised = Prepared.objects.create(
        owner=user,
        recommendation=source.recommendation,
        recommendation_version=source.recommendation_version,
        dataset=source.dataset,
        name=source.name,
        choices=choices,
        connection_refs=source.connection_refs,
        connection_versions=source.connection_versions,
        payload=payload,
        intent=intent,
        summary=summary,
    )
    revised.intent["idempotency_key"] = "edited-" + revised.id.hex
    revised.summary["draft_sha256"] = digest(
        {
            "payload": payload,
            "intent": revised.intent,
            "workload_identity": summary["workload_identity"],
        }
    )
    revised.save(update_fields=["intent", "summary"])
    job = enqueue(user, "prepare", revised, revised.connection_refs, "edit:" + revised.id.hex)
    write_artifact(job, "request.json", payload)
    write_artifact(job, "intent.json", revised.intent)
    if payload.get("workload_manifest"):
        write_artifact(job, "workload.json", payload["workload_manifest"])
    Prepared.objects.filter(pk=source.pk, approved_at__isnull=True).update(invalidated=True)
    return revised


@transaction.atomic
def clone_failed_study(user, identifier, submission_key):
    """Create an unapproved draft without mutating an executed request."""
    from .models import Job
    from .services import binding, connections
    from .study_status import study_outcome

    job = owned(Job, user, identifier)
    if job.kind != "study" or job.state not in {"failed", "cancelled", "completed"}:
        raise ValueError("Only terminal runs with failures can be rerun.")
    if job.state == "completed" and study_outcome(job) not in {
        "partial_success", "no_valid_results"
    }:
        raise ValueError("This run has no failed configurations to rerun.")
    if not isinstance(submission_key, str) or not 1 <= len(submission_key) <= 100:
        raise ValueError("A submission key is required.")
    key = "rerun:" + digest({"job": str(job.id), "submission": submission_key})
    previous = Job.objects.filter(owner=user, dedupe_key=key).first()
    if previous:
        return owned(Prepared, user, previous.target_id)
    source = owned(Prepared, user, job.target_id)
    if not source.digest or source.digest != digest(binding(source)):
        raise ValueError("The original request failed its integrity check.")
    if source.choices.get("workload_mode") != "sharegpt":
        raise ValueError("Random workloads were removed. Use Configure to select a dataset.")
    if source.recommendation.version != source.recommendation_version:
        raise ValueError("The selected model changed. Prepare a new configuration.")
    found = connections(user, source.connection_refs)
    revised = Prepared.objects.create(
        owner=user, recommendation=source.recommendation,
        recommendation_version=source.recommendation_version,
        dataset=source.dataset, name=source.name,
        choices=Choices.model_validate(source.choices).model_dump(),
        connection_refs=copy.deepcopy(source.connection_refs),
        connection_versions={kind: connection.version for kind, connection in found.items()},
        payload=copy.deepcopy(source.payload), intent=copy.deepcopy(source.intent),
        summary=copy.deepcopy(source.summary),
    )
    revised.intent["idempotency_key"] = "rerun-" + revised.id.hex
    revised.summary.update(rerun_of=str(job.id), provider_verification="pending")
    revised.summary["draft_sha256"] = digest({
        "payload": revised.payload, "intent": revised.intent,
        "workload_identity": revised.summary["workload_identity"],
    })
    revised.save(update_fields=["intent", "summary"])
    preparation = enqueue(user, "prepare", revised, revised.connection_refs, key)
    for name, content in (("request.json", revised.payload), ("intent.json", revised.intent),
                          ("workload.json", revised.payload["workload_manifest"])):
        write_artifact(preparation, name, content)
    return revised
