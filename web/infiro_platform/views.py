"""Session and CSRF-protected private API. No resource is fetched unscoped."""

import functools
import json
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import IntegrityError, transaction
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from pydantic import ValidationError as InputError

from .contracts import Choices, DatasetConfig, DirectModel, Requirements
from .models import Artifact, AuthThrottle, Connection, Dataset, Job, Prepared, Recommendation
from .naming import (
    model_id,
    recommendation_name,
    study_name,
    validate_name,
    validate_recommendation_name,
)
from .security import PROVIDERS, digest, private_dir, seal, validate_connection
from .services import TERMINAL, approve, connections, enqueue, owned
from .study_status import display_progress, study_outcome, trial_failures


def body(request, max_bytes=None):
    try:
        raw = request.read(max_bytes + 1) if max_bytes else request.body
        if max_bytes and len(raw) > max_bytes:
            raise ValueError("Request exceeds 16 MiB.")
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError("Send a JSON object.") from None
    if not isinstance(result, dict):
        raise ValueError("Send a JSON object.")
    return result


def api(methods):
    def decorate(fn):
        @require_http_methods(methods)
        @functools.wraps(fn)
        def wrapped(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({"error": "Sign in to your private workspace."}, status=401)
            try:
                return fn(request, *args, **kwargs)
            except ModuleNotFoundError as exc:
                if exc.name != "llm_optimizer" and not (exc.name or "").startswith(
                    "llm_optimizer."
                ):
                    raise
                return JsonResponse({"error": (
                    "Optimization dependencies are unavailable. Install the configured "
                    "llm-optimizer package in both the web and worker environments, then retry."
                )}, status=503)
            except InputError as exc:
                return JsonResponse(
                    {
                        "error": "Please check the form values.",
                        "fields": [
                            {"field": ".".join(str(v) for v in e["loc"]), "message": e["msg"]}
                            for e in exc.errors(
                                include_input=False, include_context=False, include_url=False
                            )
                        ],
                    },
                    status=400,
                )
            except (ValueError, ValidationError) as exc:
                return JsonResponse({"error": str(exc)}, status=400)
            except Http404:
                return JsonResponse({"error": "Resource not found in your workspace."}, status=404)
            except ImproperlyConfigured:
                return JsonResponse(
                    {
                        "error": "Configure the server's persistent credential encryption key "
                        "before saving connections."
                    },
                    status=503,
                )
            except IntegrityError:
                return JsonResponse(
                    {
                        "error": "This request is already being processed. "
                        "Refresh to see saved progress."
                    },
                    status=409,
                )

        return wrapped

    return decorate


@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def auth(request, action="login"):
    if request.method == "GET":
        if request.user.is_authenticated:
            return redirect("/app")
        return render(request, "infiro_platform/login.html", {"register": action == "register"})
    try:
        data = body(request) if request.content_type == "application/json" else request.POST
    except ValueError:
        return JsonResponse({"error": "Send a valid sign-in form."}, status=400)
    if action == "logout":
        logout(request)
        return JsonResponse({"redirect": "/app/login"})
    now = timezone.now()
    # Throttle both login and registration, without persisting passwords or names.
    key = digest({"ip": request.META.get("REMOTE_ADDR", "unknown")})
    with transaction.atomic():
        throttle, _ = AuthThrottle.objects.get_or_create(key=key, defaults={"window": now})
        if throttle.window < now - timedelta(minutes=10):
            throttle.window, throttle.attempts = now, 0
        if throttle.attempts >= 30:
            return JsonResponse(
                {"error": "Too many attempts. Try again in ten minutes."}, status=429
            )
        throttle.attempts += 1
        throttle.save()
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or len(username) > 150 or len(password) > 1024:
        return render(
            request,
            "infiro_platform/login.html",
            {"register": action == "register", "error": "Enter a valid username and password."},
            status=400,
        )
    if action == "register":
        try:
            validate_password(password, User(username=username))
            User.objects.create_user(username=username, password=password)
        except (ValidationError, IntegrityError):
            return render(
                request,
                "infiro_platform/login.html",
                {
                    "register": True,
                    "error": "Choose an available username and a strong password "
                    "of at least 12 characters.",
                },
                status=400,
            )
    user = authenticate(request, username=username, password=password)
    if user is None:
        return render(
            request,
            "infiro_platform/login.html",
            {"register": action == "register", "error": "Unable to sign in with those details."},
            status=400,
        )
    login(request, user)
    return redirect("/app")


@ensure_csrf_cookie
@require_http_methods(["GET"])
def shell(request, path=""):
    if not request.user.is_authenticated:
        return redirect("/app/login")
    import re
    import secrets

    request.csp_nonce = secrets.token_urlsafe(24)
    target = settings.FRONTEND_DIST / "app" / "index.html"
    if not target.is_file():
        return HttpResponse("Rebuild the private workspace with npm run build in web/.", status=503)
    # Astro's hydration bootstrap is inline. A per-response nonce permits only
    # these server-owned scripts, retaining a strict private-app script policy.
    html = re.sub(
        r"<script(?=[\s>])", '<script nonce="' + request.csp_nonce + '"', target.read_text()
    )
    return HttpResponse(html, content_type="text/html; charset=utf-8")


def connection_view(c):
    return {
        "id": str(c.id),
        "kind": c.kind,
        "name": c.name,
        "metadata": c.metadata,
        "version": c.version,
        "validation": c.validation,
        "validated_at": c.validated_at,
        "credentials": {
            key: "••••••••"
            for key in (
                {"private_key"}
                if c.metadata.get("auth_method") == "service_account"
                else {"iam_token"}
                if c.kind == "nebius_jobs"
                else PROVIDERS[c.kind]["required"]
            )
        },
    }


def dataset_view(d):
    return {
        "id": str(d.id),
        "name": d.name,
        "config": d.config,
        "status": d.status,
        "identity": d.identity,
        "preview": d.preview,
        "error": d.error,
        "created_at": d.created_at,
    }


def recommendation_view(r):
    # Raw provider source texts and paths are never needed by the browser.
    candidates = []
    for row in r.report.get("recommendations", []):
        candidate = {
            k: row.get(k)
            for k in (
                "model_id",
                "revision",
                "reasoning",
                "hf_score",
                "final_score",
                "why_it_fits",
                "limitations",
                "task_fit",
                "runtime_assessment",
            )
        }
        candidate["evidence"] = [
            {
                "source_url": e.get("source_url"),
                "title": e.get("title", "Evidence source"),
                "status": e.get("status", "unknown"),
            }
            for e in row.get("evidence", [])
            if isinstance(e, dict)
        ]
        candidates.append(candidate)
    return {
        "id": str(r.id),
        "name": recommendation_name(r),
        "requirements": r.requirements,
        "connection_refs": r.connection_refs,
        "status": r.status,
        "recommendations": candidates,
        "selected": r.selected,
        "version": r.version,
        "validation": r.report.get("validation", {}),
        "data_origin": r.report.get("data_origin", "provider_research"),
        "selection_source": r.report.get("selection_source", "recommendation"),
        "created_at": r.created_at,
    }


def prepared_view(p):
    return {
        "id": str(p.id),
        "name": study_name(p),
        "model_id": model_id(p),
        "recommendation_id": str(p.recommendation_id),
        "recommendation_name": recommendation_name(p.recommendation),
        "selection_source": p.recommendation.report.get("selection_source", "recommendation"),
        "dataset_id": str(p.dataset_id) if p.dataset_id else None,
        "choices": p.choices,
        "status": p.status,
        "summary": p.summary,
        "digest": p.digest,
        "invalidated": p.invalidated,
        "approved_at": p.approved_at,
        "configuration_files": [
            {"id": str(a.id), "name": a.name,
             "download_url": "/app/api/artifacts/" + str(a.id)}
            for a in Artifact.objects.filter(
                owner=p.owner, job__owner=p.owner, job__kind="prepare", job__target_id=p.id,
                name__in=("request.json", "intent.json", "workload.json"),
            )
        ],
        "connection_refs": p.connection_refs,
        "created_at": p.created_at,
    }


def job_view(j):
    from .history import visible_result

    result = visible_result(j)
    state = j.state
    if j.kind == "study" and state == "queued" and j.controller_run:
        state = j.progress.get("state", state)
    return {
        "id": str(j.id),
        "kind": j.kind,
        "target_id": str(j.target_id),
        "state": state,
        "queue_state": j.state,
        "outcome": study_outcome(j),
        "trial_failures": trial_failures(result) if j.kind == "study" else [],
        "attempts": j.attempts,
        "cancel_requested": j.cancel_requested,
        "error": j.error,
        "progress": display_progress(j),
        "result": result,
        "created_at": j.created_at,
        "updated_at": j.updated_at,
        "artifacts": [
            {
                "id": str(a.id),
                "name": a.name,
                "size": a.size,
                "sha256": a.sha256,
                "download_url": "/app/api/artifacts/" + str(a.id),
            }
            for a in Artifact.objects.filter(job=j, owner=j.owner)
        ],
    }


@api(["GET"])
def bootstrap(request):
    from hf_model_watch.optimization.dspark import capabilities
    from hf_model_watch.optimization.objectives import objective_options

    user = request.user
    return JsonResponse(
        {
            "user": {"username": user.username},
            "optimization_objectives": objective_options(),
            "connections": [
                connection_view(c) for c in Connection.objects.filter(owner=user).order_by("name")
            ],
            "datasets": [
                dataset_view(d)
                for d in Dataset.objects.filter(owner=user).order_by("-created_at")[:100]
            ],
            "recommendations": [
                recommendation_view(r)
                for r in Recommendation.objects.filter(owner=user, deleted_at__isnull=True)
                .order_by("-created_at")[:50]
            ],
            "prepared": [
                prepared_view(p)
                for p in Prepared.objects.filter(owner=user, deleted_at__isnull=True)
                .order_by("-created_at")[:100]
            ],
            "jobs": [
                job_view(j) for j in Job.objects.filter(owner=user, deleted_at__isnull=True)
                .order_by("-created_at")[:100]
            ],
            "live_enabled": bool(settings.INFIRO_LIVE_ENABLED and settings.INFIRO_PROVIDER_FACTORY),
            "dspark": capabilities(
                settings.INFIRO_DSPARK_COMPATIBILITY, settings.INFIRO_SERVING_IMAGES
            ),
            "tracking_configuration": {
                "project": settings.INFIRO_WANDB.get("project"),
                "entity": settings.INFIRO_WANDB.get("entity"),
            },
            "limits": {
                "download_mib": 128,
                "rows": 10000,
                "source_tokens": 2000000,
                "requested_tokens": 2000000,
                "sample_count": 1000,
                "memory_gib": 6,
                "disk_mib": 256,
                "operation_seconds": 300,
            },
        }
    )


def active_connection(user, identifier):
    for job in Job.objects.filter(owner=user).exclude(state__in=TERMINAL):
        if str(identifier) in job.connection_refs.values():
            raise ValueError(
                "This connection is in use. Finish or cancel its active operations "
                "before replacing or deleting it."
            )


@api(["POST"])
def create_connection(request):
    data = body(request)
    name = str(data.get("name", "")).strip()
    if not 1 <= len(name) <= 80:
        raise ValueError("Connection name must be 1–80 characters.")
    metadata = validate_connection(
        data.get("kind"), data.get("metadata", {}), data.get("credentials", {})
    )
    c = Connection(owner=request.user, name=name, kind=data["kind"], metadata=metadata)
    c.ciphertext = seal(c, data["credentials"])
    c.save()
    return JsonResponse(connection_view(c), status=201)


@api(["GET", "PUT", "DELETE", "POST"])
@transaction.atomic
def connection_detail(request, identifier):
    c = owned(Connection, request.user, identifier)
    if request.method == "GET":
        return JsonResponse(connection_view(c))
    if request.method == "POST":
        active = (
            Job.objects.filter(owner=request.user, kind="validate", target_id=c.id)
            .exclude(state__in=TERMINAL)
            .first()
        )
        if active:
            return JsonResponse(job_view(active), status=202)
        from uuid import uuid4

        job = enqueue(
            request.user,
            "validate",
            c,
            {c.kind: str(c.id)},
            "validate:" + str(c.id) + ":" + str(c.version) + ":" + uuid4().hex,
        )
        return JsonResponse(job_view(job), status=202)
    replacement = body(request) if request.method == "PUT" else None
    new_metadata = (
        validate_connection(
            c.kind, replacement.get("metadata", c.metadata), replacement.get("credentials", {})
        )
        if replacement
        else None
    )
    with transaction.atomic():
        renewals = []
        if request.method == "PUT" and c.kind == "nebius_jobs" and new_metadata == c.metadata:
            active = [
                j
                for j in Job.objects.filter(owner=request.user).exclude(state__in=TERMINAL)
                if str(c.id) in j.connection_refs.values()
            ]
            if all(j.kind == "study" and j.controller_run for j in active):
                renewals = active
            else:
                active_connection(request.user, c.id)
        else:
            active_connection(request.user, c.id)
        for p in Prepared.objects.filter(owner=request.user, invalidated=False):
            if str(c.id) in p.connection_refs.values():
                Prepared.objects.filter(pk=p.id).update(invalidated=True)
        if request.method == "DELETE":
            c.delete()
            return JsonResponse({"deleted": True})
        c.metadata = new_metadata
        c.ciphertext = seal(c, replacement["credentials"])
        c.version += 1
        c.validation = "unvalidated"
        c.validated_at = None
        c.save()
        # Refresh only the IAM transport credential of already allocated jobs.
        # Their approved requests and original review snapshots remain immutable.
        for job in renewals:
            job.connection_versions[c.kind] = c.version
            job.save(update_fields=["connection_versions", "updated_at"])
    return JsonResponse(connection_view(c))


@api(["POST"])
@transaction.atomic
def create_recommendation(request):
    data = body(request)
    name = validate_recommendation_name(data.get("name", ""))
    requirements = Requirements.model_validate(data.get("requirements", {})).model_dump()
    refs = data.get("connection_refs", {})
    found = connections(request.user, refs, required=("nebius_inference",))
    key = "discovery:" + digest(
        {
            "requirements": requirements,
            "versions": {k: c.version for k, c in found.items()},
            "refs": refs,
            "nonce": str(data.get("submission_key", ""))[:100],
        }
    )
    previous = Job.objects.filter(owner=request.user, dedupe_key=key).first()
    if previous:
        return JsonResponse(
            {
                "recommendation": recommendation_view(
                    owned(Recommendation, request.user, previous.target_id)
                ),
                "job": job_view(previous),
            },
            status=200,
        )
    rec = Recommendation.objects.create(
        owner=request.user, name=name, requirements=requirements, connection_refs=refs
    )
    job = enqueue(request.user, "discover", rec, refs, key)
    return JsonResponse(
        {"recommendation": recommendation_view(rec), "job": job_view(job)}, status=202
    )


@api(["GET", "POST", "PATCH", "DELETE"])
@transaction.atomic
def recommendation_detail(request, identifier):
    if request.method == "DELETE":
        from .history import delete_history
        delete_history(request.user, Recommendation, identifier)
        return JsonResponse({"deleted": True})
    rec = owned(Recommendation, request.user, identifier)
    if request.method == "GET":
        return JsonResponse(recommendation_view(rec))
    data = body(request)
    if request.method == "PATCH":
        if set(data) != {"name"}:
            raise ValueError("Only the recommendation name can be renamed.")
        rec.name = validate_recommendation_name(data["name"])
        rec.save(update_fields=["name"])
        return JsonResponse(recommendation_view(rec))
    selected = next(
        (
            r
            for r in rec.report.get("recommendations", [])
            if r.get("model_id") == data.get("model_id")
        ),
        None,
    )
    if rec.status != "recommended" or selected is None:
        raise ValueError("Choose a model from this recommendation.")
    revision = selected.get("revision") or (selected.get("runtime_assessment") or {}).get(
        "revision"
    )
    import re

    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision or ""):
        raise ValueError(
            "This candidate has no verified commit. Rerun discovery before preparation."
        )
    rec.selected = {"model_id": selected["model_id"], "revision": revision}
    rec.version += 1
    rec.save()
    Prepared.objects.filter(owner=request.user, recommendation=rec).update(invalidated=True)
    return JsonResponse(recommendation_view(rec))


@api(["POST"])
@transaction.atomic
def create_dataset(request):
    data = body(request)
    config = DatasetConfig.model_validate(data.get("config", {})).model_dump()
    from .dataset_source import require_optimizer

    require_optimizer()
    refs = data.get("connection_refs", {})
    found = connections(request.user, refs)
    name = str(data.get("name") or config["repo_id"])[:100]
    key = "dataset:" + digest(
        {"config": config, "refs": refs, "versions": {k: c.version for k, c in found.items()}}
    )
    previous = Job.objects.filter(owner=request.user, dedupe_key=key).first()
    if previous:
        return JsonResponse(
            {
                "dataset": dataset_view(owned(Dataset, request.user, previous.target_id)),
                "job": job_view(previous),
            },
            status=200,
        )
    dataset = Dataset.objects.create(
        owner=request.user, name=name, config=config, connection_refs=refs, status="validating"
    )
    job = enqueue(request.user, "dataset", dataset, refs, key)
    return JsonResponse({"dataset": dataset_view(dataset), "job": job_view(job)}, status=202)


@api(["GET"])
def dataset_detail(request, identifier):
    return JsonResponse(dataset_view(owned(Dataset, request.user, identifier)))


@api(["POST"])
@transaction.atomic
def create_prepared(request):
    data = body(request)
    name = validate_name(data.get("name", ""))
    rec = owned(Recommendation, request.user, data.get("recommendation_id"))
    if not rec.selected:
        raise ValueError("Select a recommended model first.")
    choices = Choices.model_validate(data.get("choices", {})).model_dump()
    if choices.get("speculative_method") == "dspark":
        from hf_model_watch.optimization.dspark import resolve

        resolve(
            rec.selected["model_id"], rec.selected["revision"], choices["frameworks"],
            settings.INFIRO_SERVING_IMAGES, settings.INFIRO_DSPARK_COMPATIBILITY,
        )
    dataset = owned(Dataset, request.user, data["dataset_id"]) if data.get("dataset_id") else None
    if dataset and dataset.status != "ready":
        raise ValueError("Validate and freeze the dataset first.")
    refs = data.get("connection_refs", rec.connection_refs)
    found = connections(
        request.user,
        refs,
        required=("nebius_inference",)
        if choices["dry_run"]
        else ("nebius_inference", "nebius_jobs", "storage"),
    )
    key = "prepare:" + digest(
        {
            "rec": str(rec.id),
            "version": rec.version,
            "choices": choices,
            "dataset": str(dataset.id) if dataset else None,
            "connections": {k: [str(c.id), c.version] for k, c in found.items()},
            "submission_key": str(data.get("submission_key", ""))[:100],
        }
    )
    previous = Job.objects.filter(owner=request.user, dedupe_key=key).first()
    if previous:
        return JsonResponse(
            {
                "prepared": prepared_view(owned(Prepared, request.user, previous.target_id)),
                "job": job_view(previous),
            },
            status=200,
        )
    # All older unlaunched reviews are invalid once the saved configuration changes.
    Prepared.objects.filter(owner=request.user, recommendation=rec).update(invalidated=True)
    p = Prepared.objects.create(
        name=name,
        owner=request.user,
        recommendation=rec,
        dataset=dataset,
        choices=choices,
        connection_refs=refs,
        connection_versions={k: c.version for k, c in found.items()},
        recommendation_version=rec.version,
    )
    if not p.name:
        p.name = study_name(p)
        p.save(update_fields=["name"])
    job = enqueue(request.user, "prepare", p, refs, key)
    return JsonResponse({"prepared": prepared_view(p), "job": job_view(job)}, status=202)


@api(["POST"])
@transaction.atomic
def select_direct_model(request):
    data = body(request)
    model = DirectModel.model_validate(data.get("model", {})).model_dump()
    refs = data.get("connection_refs", {})
    connections(request.user, refs)
    key = "model:" + digest({"model": model, "refs": refs,
                              "submission_key": str(data.get("submission_key", ""))[:100]})
    previous = Job.objects.filter(owner=request.user, dedupe_key=key).first()
    if previous:
        rec = owned(Recommendation, request.user, previous.target_id)
        return JsonResponse({"recommendation": recommendation_view(rec), "job": job_view(previous)})
    rec = Recommendation.objects.create(owner=request.user, name=model["model_id"],
        requirements={"request": "Direct model selection"}, connection_refs=refs,
        status="queued", report={"selection_source": "direct", "data_origin": "user_selected",
                                  "requested_model": model})
    job = enqueue(request.user, "model", rec, refs, key)
    return JsonResponse(
        {"recommendation": recommendation_view(rec), "job": job_view(job)}, status=202)


@api(["GET", "POST"])
def edit_prepared_request(request, identifier):
    from .request_editing import revise_request

    if request.method == "GET":
        prepared = owned(Prepared, request.user, identifier)
        return JsonResponse({"request": prepared.payload, "digest": prepared.digest})
    data = body(request, max_bytes=16 * 1024**2)
    prepared = revise_request(request.user, identifier, data.get("digest"), data.get("request"))
    return JsonResponse({"prepared": prepared_view(prepared)}, status=201)


@api(["GET", "POST", "PATCH", "DELETE"])
def prepared_detail(request, identifier):
    if request.method == "DELETE":
        from .history import delete_history
        delete_history(request.user, Prepared, identifier)
        return JsonResponse({"deleted": True})
    if request.method == "PATCH":
        p = owned(Prepared, request.user, identifier)
        name = validate_name(body(request).get("name"))
        p.name = name
        Prepared.objects.filter(pk=p.pk, owner=request.user).update(name=name)
        return JsonResponse(prepared_view(p))
    if request.method == "GET":
        return JsonResponse(prepared_view(owned(Prepared, request.user, identifier)))
    data = body(request)
    job = approve(request.user, identifier, data.get("digest"), data.get("approved"))
    return JsonResponse(job_view(job), status=202)


@api(["GET", "POST", "DELETE"])
def job_detail(request, identifier):
    if request.method == "DELETE":
        from .history import delete_history
        delete_history(request.user, Job, identifier)
        return JsonResponse({"deleted": True})
    job = owned(Job, request.user, identifier)
    if request.method == "POST" and job.state not in TERMINAL:
        Job.objects.filter(pk=job.pk, owner=request.user).update(
            cancel_requested=True,
            state="queued" if job.state == "attention_required" else job.state,
            available_at=timezone.now(),
        )
        job.refresh_from_db()
    return JsonResponse(job_view(job))


@api(["GET"])
def study_diagnostics(request, identifier):
    job = owned(Job, request.user, identifier)
    if job.kind != "study" or not job.controller_run:
        raise ValueError("Logs are available only for launched runs.")
    from .study_logs import saved_logs

    try:
        result = saved_logs(job)
    except Exception:
        # SDK errors can contain signed URLs or credentials. Do not echo them.
        return JsonResponse({"error": "Saved logs could not be read. Check the storage connection "
                              "and retry."}, status=503)
    response = JsonResponse(result)
    response["Cache-Control"] = "no-store"
    return response


@api(["POST"])
@transaction.atomic
def retry_job(request, identifier):
    job = owned(Job, request.user, identifier)
    if (
        job.kind == "study"
        and (job.state != "attention_required" or not job.controller_run)
        or job.kind != "study"
        and job.state not in {"failed", "cancelled"}
    ):
        raise ValueError(
            "Only failed or cancelled operations can be retried; "
            "a new run requires fresh preparation and approval."
        )
    found = connections(request.user, job.connection_refs)
    if {k: c.version for k, c in found.items()} != job.connection_versions:
        raise ValueError(
            "Connections changed. Submit a new configuration using the current versions."
        )
    Job.objects.filter(pk=job.pk).update(
        state="queued",
        attempts=0,
        cancel_requested=False,
        lease_until=None,
        available_at=timezone.now(),
        error="",
    )
    model = {"model": Recommendation, "discover": Recommendation,
             "dataset": Dataset, "prepare": Prepared}.get(job.kind)
    if model:
        target = owned(model, request.user, job.target_id)
        if isinstance(target, Prepared) and target.invalidated:
            raise ValueError("This review is stale. Prepare a new configuration.")
        if not (isinstance(target, Prepared) and target.digest):
            model.objects.filter(pk=job.target_id, owner=request.user).update(
                status="validating" if job.kind == "dataset" else "queued"
            )
    job.refresh_from_db()
    return JsonResponse(job_view(job), status=202)


@api(["GET"])
def artifact_download(request, identifier):
    artifact = owned(Artifact, request.user, identifier)
    job = owned(Job, request.user, artifact.job_id)
    root = private_dir(request.user.id, job.id)
    path = (root / artifact.relative_path).resolve()
    if path.parent != root.resolve() or not path.is_file() or path.is_symlink():
        raise Http404
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
        return JsonResponse({"error": "Artifact integrity check failed."}, status=409)
    return FileResponse(
        path.open("rb"), as_attachment=True, filename=artifact.name, content_type="application/json"
    )


@api(["DELETE"])
def trial_detail(request, identifier):
    from .history import delete_trial
    delete_trial(request.user, identifier, body(request).get("trial_key"))
    return JsonResponse({"deleted": True})


@api(["POST"])
def rerun_job(request, identifier):
    from .request_editing import clone_failed_study

    prepared = clone_failed_study(request.user, identifier, body(request).get("submission_key"))
    return JsonResponse({"prepared": prepared_view(prepared)}, status=202)
