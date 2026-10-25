"""Nebius SDK adapter with immutable SecretStash references, never inline secrets.

Uses the official AI JobSpec.EnvironmentVariable.mysterybox_secret contract.
Credentials must already exist in the user's project in pinned secret versions.
The adapter verifies their project and values before a job can be submitted.
No secret-management writes, CLI commands or profile authentication are used.
"""

import hmac
import os
import re
from datetime import timedelta
from functools import wraps

from llm_optimizer.remote.adapters import S3Artifacts
from llm_optimizer.remote.contracts import OptimizationRequest

RPC = {
    "timeout": 30,
    "auth_timeout": 30,
    "retries": 0,
    "auth_options": {"token_renew_report_error": "true", "max_fetch_token_retries": "0"},
}


def observed_provider_call(method):
    """Retain safe errors even when the shared controller catches exceptions."""

    @wraps(method)
    def call(self, *args, **kwargs):
        try:
            result = method(self, *args, **kwargs)
            progress = getattr(self.job, "progress", {})
            active = progress.get("provider_diagnostics", {})
            scopes = progress.get("provider_error_scopes", {})
            scope = args[0] if method.__name__ in {"status", "cancel"} and args else None
            recovered = None
            if method.__name__ in scopes and scopes[method.__name__] == scope:
                recovered = active.pop(method.__name__, None)
                scopes.pop(method.__name__, None)
            if recovered:
                history = progress.setdefault("recovered_provider_diagnostics", {})
                history[method.__name__] = recovered
                if active:
                    from .diagnostics import failure_summary
                    progress["provider_error"] = failure_summary(next(iter(active.values())))
                else:
                    progress.pop("provider_error", None)
                    progress.pop("provider_diagnostics", None)
            return result
        except Exception as exc:
            from .diagnostics import failure_details, failure_summary, submission_validation_details

            details = failure_details(exc)
            metadata = getattr(self, "_submission_metadata", None)
            if method.__name__ == "submit" and metadata is not None:
                details.update(submission_validation_details(exc, metadata))
            progress = getattr(self.job, "progress", {})
            progress.setdefault("provider_diagnostics", {})[method.__name__] = details
            progress["provider_error"] = failure_summary(details)
            progress.setdefault("provider_error_scopes", {})[method.__name__] = (
                args[0] if method.__name__ in {"status", "cancel"} and args else None
            )
            self.job.progress = progress
            raise

    return call


def pin_secret_references(found):
    """Resolve mutable secret references before the review is hashed and approved."""
    from nebius.api.nebius.mysterybox.v1 import (
        GetSecretRequest,
        GetSecretVersionRequest,
        SecretServiceClient,
        SecretVersionServiceClient,
    )

    from .nebius_auth import make_sdk

    settings = {kind: dict(connection.metadata) for kind, connection in found.items()}
    sdk = make_sdk(
        found["nebius_jobs"].metadata,
        {
            key: os.environ[name]
            for key, name in {
                "iam_token": "NEBIUS_IAM_TOKEN",
                "private_key": "NEBIUS_PRIVATE_KEY",
            }.items()
            if name in os.environ
        },
    )
    try:
        secrets = SecretServiceClient(sdk)
        versions = SecretVersionServiceClient(sdk)
        for kind in ("huggingface", "storage", "wandb"):
            if kind not in found:
                continue
            reference = settings[kind]["secret_version_id"]
            # Resolve by resource type using provider reads, not an assumed ID prefix.
            try:
                version = versions.get(GetSecretVersionRequest(id=reference), **RPC).wait()
                secret = secrets.get(GetSecretRequest(id=version.metadata.parent_id), **RPC).wait()
            except Exception as error:
                from nebius.aio.service_error import RequestError

                if not isinstance(error, RequestError) or error.status.code.name not in {
                    "NOT_FOUND",
                    "INVALID_ARGUMENT",
                }:
                    raise
                if any(
                    item.service == "token-service" or item.code.startswith("Jwt")
                    for item in getattr(error.status, "service_errors", [])
                ):
                    # IAM failed before a SecretStash resource lookup. Trying a
                    # different resource type cannot repair authentication.
                    raise
                secret = secrets.get(GetSecretRequest(id=reference), **RPC).wait()
                reference = secret.spec.primary_version_id
                if not reference:
                    raise ValueError("The SecretStash secret has no primary version.") from None
                version = versions.get(GetSecretVersionRequest(id=reference), **RPC).wait()
            if secret.metadata.parent_id != found["nebius_jobs"].metadata["project_id"]:
                raise ValueError("SecretStash secret belongs to a different Nebius project.")
            if version.metadata.parent_id != secret.metadata.id:
                raise ValueError("SecretStash version does not belong to the selected secret.")
            settings[kind]["secret_version_id"] = version.metadata.id
        return settings
    finally:
        sdk.sync_close()


def validate_live_configuration(prepared, found, images):
    if "nebius_jobs" not in found or "storage" not in found:
        raise ValueError("Live studies require separate Nebius jobs and storage connections.")
    for framework in prepared.choices["frameworks"]:
        if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", images.get(framework, "")):
            raise ValueError("Live worker images must be configured and pinned by SHA-256 digest.")
    for kind in ("storage", "huggingface"):
        if kind in found and not found[kind].metadata.get("secret_version_id"):
            raise ValueError(
                "Live worker credentials require immutable SecretStash secret version references."
            )


class NebiusSDKProvider:
    def __init__(self, job, prepared, found, *, sdk=None, payload_storage_factory=None):
        from nebius.api.nebius.ai.v1 import JobServiceClient

        from .nebius_auth import make_sdk

        self.job = job
        self.prepared = prepared
        self.found = found
        self.payload_storage_factory = payload_storage_factory
        for kind in ("huggingface", "storage"):
            pinned = prepared.summary.get("provider_settings", {}).get(kind)
            if kind in found and pinned:
                from copy import copy

                self.found = dict(self.found)
                self.found[kind] = copy(found[kind])
                self.found[kind].metadata = dict(pinned)
        self.project_id = found["nebius_jobs"].metadata["project_id"]
        values = {
            key: os.environ[env_key]
            for key, env_key in {
                "iam_token": "NEBIUS_IAM_TOKEN",
                "private_key": "NEBIUS_PRIVATE_KEY",
            }.items()
            if env_key in os.environ
        }
        self.sdk = sdk or make_sdk(found["nebius_jobs"].metadata, values)
        self.client = JobServiceClient(self.sdk)
        self._verified = False

    def name(self, run_id, framework):
        return "infiro-" + self.job.id.hex + "-" + framework

    def ownership_labels(self):
        # Recognize the previous adapter's labels during reconciliation only.
        # New job creation deliberately omits custom labels, as the CLI does.
        return {"infirojob": "job" + self.job.id.hex, "infirouser": "user" + str(self.job.owner_id)}

    def _verify_secret(self, connection, expected, *, required_keys=()):
        from nebius.api.nebius.mysterybox.v1 import (
            GetPayloadRequest,
            GetSecretRequest,
            GetSecretVersionRequest,
            PayloadServiceClient,
            SecretServiceClient,
            SecretVersionServiceClient,
        )

        version_id = connection.metadata["secret_version_id"]
        version = (
            SecretVersionServiceClient(self.sdk)
            .get(GetSecretVersionRequest(id=version_id), **RPC)
            .wait()
        )
        secret = (
            SecretServiceClient(self.sdk)
            .get(GetSecretRequest(id=version.metadata.parent_id), **RPC)
            .wait()
        )
        if secret.metadata.parent_id != self.project_id:
            raise ValueError("Worker secret version belongs to a different Nebius project.")
        payload = (
            PayloadServiceClient(self.sdk)
            .get(GetPayloadRequest(version_id=version_id), **RPC)
            .wait()
        )
        values = {entry.key: entry.string_value for entry in payload.data}
        if (
            payload.version_id != version_id
            or any(not values.get(key) for key in required_keys)
            or any(
                not hmac.compare_digest(values.get(key, ""), value)
                for key, value in expected.items()
            )
        ):
            raise ValueError(
                "Worker secret version does not match the reviewed connection credentials."
            )

    @observed_provider_call
    def verify_secrets(self):
        if self._verified:
            return
        keys = {
            "storage": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"),
            "huggingface": ("HF_TOKEN",),
        }
        for kind, names in keys.items():
            if kind in self.found:
                expected = {key: os.environ[key] for key in names if os.getenv(key)}
                self._verify_secret(self.found[kind], expected)
        tracking = self.prepared.summary.get("tracking", {})
        if tracking:
            from types import SimpleNamespace

            if not tracking.get("project") or not tracking.get("secret_version_id"):
                raise ValueError(
                    "Shared W&B tracking requires a project and pinned SecretStash version."
                )
            self._verify_secret(
                SimpleNamespace(metadata={"secret_version_id": tracking["secret_version_id"]}),
                {},
                required_keys=("WANDB_API_KEY",),
            )
        self._verified = True

    def build_spec(self, payload):
        from nebius.api.nebius.ai.v1 import (
            FollowsSpotPriceSpec,
            JobSpec,
            OnDemandSpec,
            PricingModelSpec,
            SpotPricingPolicySpec,
        )
        from nebius.api.nebius.compute.v1 import DiskSpec

        request = OptimizationRequest.parse(payload["request"]).to_dict()
        validate_live_configuration(self.prepared, self.found, request["images"])
        import json

        from .payload_transport import REFERENCE_ENV, bootstrap_args, payload_blob

        _, reference = payload_blob(payload, self.prepared, self.job)
        environment = [JobSpec.EnvironmentVariable(name=REFERENCE_ENV, value=json.dumps(reference))]
        names = {
            "storage": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"),
            "huggingface": ("HF_TOKEN",),
        }
        for kind, variables in names.items():
            if kind in self.found:
                for name in variables:
                    if os.getenv(name):
                        environment.append(
                            JobSpec.EnvironmentVariable(
                                name=name,
                                mysterybox_secret=JobSpec.MysteryBoxSecretRef(
                                    version_id=self.found[kind].metadata["secret_version_id"]
                                ),
                            )
                        )
        storage = self.found["storage"].metadata
        tracking = self.prepared.summary.get("tracking", {})
        if tracking:
            environment.append(JobSpec.EnvironmentVariable(
                name="WANDB_API_KEY",
                mysterybox_secret=JobSpec.MysteryBoxSecretRef(
                    version_id=tracking["secret_version_id"]
                ),
            ))
            variables = {
                "WANDB_PROJECT": tracking["project"],
                "WANDB_MODE": "online",
                "WANDB_RUN_NAME": f"{self.job.id.hex}-{payload['framework']}",
                "WANDB_TAGS": (
                    f"infiro,user-{self.job.owner_id},study-{self.job.id.hex},"
                    f"framework-{payload['framework']}"
                ),
            }
            if tracking.get("entity"):
                variables["WANDB_ENTITY"] = tracking["entity"]
            environment.extend(
                JobSpec.EnvironmentVariable(name=name, value=value)
                for name, value in variables.items()
            )
        environment.append(
            JobSpec.EnvironmentVariable(name="AWS_DEFAULT_REGION", value=storage["region"])
        )
        if storage.get("endpoint"):
            environment.append(
                JobSpec.EnvironmentVariable(name="AWS_ENDPOINT_URL_S3", value=storage["endpoint"])
            )
        pricing = PricingModelSpec(on_demand=OnDemandSpec())
        if request["preemptible"]:
            pricing = (
                PricingModelSpec(
                    spot_pricing_policy=SpotPricingPolicySpec(id=request["spot_pricing_policy_id"])
                )
                if request["spot_pricing_policy_id"]
                else PricingModelSpec(follows_spot_price=FollowsSpotPriceSpec())
            )
        return JobSpec(
            image=request["images"][payload["framework"]],
            environment_variables=environment,
            container_command="python3",
            args=bootstrap_args(),
            platform=request["gpu_type"],
            preset=request["preset"],
            disk=JobSpec.DiskSpec(type=DiskSpec.DiskType.NETWORK_SSD, size_bytes=450 * 1024**3),
            timeout=timedelta(minutes=request["timeout_minutes"]),
            restart_attempts=0,
            preemptible=request["preemptible"],
            pricing_model=pricing,
        )

    @observed_provider_call
    def submit(self, payload):
        from nebius.api.nebius.ai.v1 import CreateJobRequest
        from nebius.api.nebius.common.v1 import ResourceMetadata

        self._submission_metadata = None
        self.verify_secrets()
        existing = self.find_by_name(payload["run_id"], payload["framework"])
        if existing:
            return existing
        spec = self.build_spec(payload)
        from django.conf import settings

        from .image_aliases import verified_alias

        # Keep the reviewed payload digest-pinned; only the provider-facing image
        # uses a verified tag to avoid Nebius' digest-reference label rejection.
        spec.image = verified_alias(spec.image, settings.INFIRO_IMAGE_ALIASES, self.sdk)
        self.stage_payload(payload)
        request = CreateJobRequest(
            metadata=ResourceMetadata(
                parent_id=self.project_id,
                name=self.name(payload["run_id"], payload["framework"]),
            ),
            spec=spec,
        )
        wire_request = CreateJobRequest.FromString(request.SerializeToString())
        labels = dict(wire_request.metadata.labels)
        self._submission_metadata = {
            "adapter_schema": "unlabelled-private-s3-v1",
            "framework": payload["framework"],
            "image_reference_length": len(wire_request.spec.image),
            "label_count": len(labels),
            "label_keys": sorted(labels),
            "all_label_keys_and_values_are_strings": all(
                isinstance(key, str) and isinstance(value, str) for key, value in labels.items()
            ),
        }
        operation = self.client.create(request, **RPC).wait()
        operation.sync_wait(timeout=60, poll_retries=0)
        if not re.fullmatch(r"aijob-[a-z0-9]+", operation.resource_id):
            raise ValueError(
                "Provider job identity is uncertain; reconcile by name before retrying."
            )
        return operation.resource_id

    def stage_payload(self, payload):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from .payload_transport import payload_blob
        from .security import private_dir

        raw, reference = payload_blob(payload, self.prepared, self.job)
        store = self.payload_storage_factory(reference["bucket"], reference["prefix"])
        name = reference["key"].rsplit("/", 1)[1]
        if store.key(payload["run_id"], payload["framework"], name) != reference["key"]:
            raise ValueError("Worker payload storage differs from its approved reference.")
        with TemporaryDirectory(dir=private_dir(self.job.owner_id, self.job.id)) as temporary:
            path = Path(temporary) / name
            path.write_bytes(raw)
            path.chmod(0o600)
            store.upload(path, payload["run_id"], payload["framework"], name)

    def _check_owned(self, remote, expected_id=None, expected_name=None):
        labels = remote.metadata.labels
        valid_names = {
            self.name(self.job.controller_run, framework)
            for framework in self.prepared.choices["frameworks"]
        }
        if "infirojob" in labels or "infirouser" in labels:
            owned = all(labels.get(key) == value for key, value in self.ownership_labels().items())
        elif "infiro-job" in labels or "infiro-user" in labels:
            # Continue reconciling/cancelling jobs created by the previous adapter.
            owned = labels.get("infiro-job") == self.job.id.hex and labels.get(
                "infiro-user"
            ) == str(self.job.owner_id)
        else:
            # New jobs omit custom labels, matching the existing CLI contract.
            # Names contain the unguessable UUID of this owner-scoped DB study.
            owned = True
        if (
            remote.metadata.parent_id != self.project_id
            or not owned
            or remote.metadata.name not in valid_names
            or (expected_name is not None and remote.metadata.name != expected_name)
            or (expected_id is not None and remote.metadata.id != expected_id)
        ):
            raise ValueError("Provider job does not belong to this private study.")
        return remote

    @observed_provider_call
    def find_by_name(self, run_id, framework):
        from grpc import StatusCode
        from nebius.aio.service_error import RequestError
        from nebius.api.nebius.ai.v1 import GetJobByNameRequest

        if (
            run_id != self.job.controller_run
            or framework not in self.prepared.choices["frameworks"]
        ):
            raise ValueError("Provider lookup is outside this study's authorized scope.")

        try:
            remote = self.client.get_by_name(
                GetJobByNameRequest(parent_id=self.project_id, name=self.name(run_id, framework)),
                **RPC,
            ).wait()
        except RequestError as exc:
            if exc.status.code == StatusCode.NOT_FOUND:
                return None
            raise
        return self._check_owned(remote, expected_name=self.name(run_id, framework)).metadata.id

    def get(self, job_id):
        from nebius.api.nebius.ai.v1 import GetJobRequest

        return self._check_owned(
            self.client.get(GetJobRequest(id=job_id), **RPC).wait(), expected_id=job_id
        )

    @observed_provider_call
    def status(self, job_id):
        from nebius.api.nebius.ai.v1 import JobStatus

        state = JobStatus.State(self.get(job_id).status.state).name.lower()
        return "failed" if state == "error" else state

    @observed_provider_call
    def cancel(self, job_id):
        from nebius.api.nebius.ai.v1 import CancelJobRequest

        self.get(job_id)
        operation = self.client.cancel(CancelJobRequest(id=job_id), **RPC).wait()
        operation.sync_wait(timeout=60, poll_retries=0)


def factory(job, prepared):
    import boto3
    from botocore.config import Config

    from .security import endpoint
    from .services import connections

    found = connections(job.owner, prepared.connection_refs, required=("nebius_jobs", "storage"))
    validate_live_configuration(prepared, found, prepared.payload["images"])
    storage = found["storage"].metadata
    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=storage["region"],
        endpoint_url=endpoint(storage["endpoint"]) if storage.get("endpoint") else None,
        config=Config(
            connect_timeout=10,
            read_timeout=20,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},
        ),
    )

    class ScopedArtifacts(S3Artifacts):
        def __init__(self, bucket, prefix):
            if bucket != prepared.payload["bucket"] or prefix != prepared.payload["prefix"]:
                raise ValueError("Storage destination differs from the approved private prefix.")
            super().__init__(bucket, prefix, client=client)

        def key(self, run_id, framework, name):
            if (
                run_id != job.controller_run
                or framework not in {*prepared.choices["frameworks"], "workload"}
                or "/" in name
                or "\\" in name
                or ".." in name
            ):
                raise ValueError("Storage artifact is outside this study's authorized scope.")
            return super().key(run_id, framework, name)

        def put_report(self, run_id, report):
            if run_id != job.controller_run:
                raise ValueError("Storage report is outside this study's authorized scope.")
            return super().put_report(run_id, report)

    return NebiusSDKProvider(
        job, prepared, found, payload_storage_factory=ScopedArtifacts
    ), ScopedArtifacts


def validate_job_connection(connection, values):
    from nebius.api.nebius.ai.v1 import JobServiceClient, ListJobsRequest

    from .nebius_auth import make_sdk

    sdk = make_sdk(connection.metadata, values)
    try:
        JobServiceClient(sdk).list(
            ListJobsRequest(parent_id=connection.metadata["project_id"], page_size=1), **RPC
        ).wait()
    finally:
        sdk.sync_close()


factory.validate_connection = validate_job_connection
