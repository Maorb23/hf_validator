import io
import json
import os
import socket
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.utils import timezone

from .contracts import Choices
from .demo import FixtureTokenizer, fixture_manifest, seed_prepared
from .models import Artifact, Connection, Dataset, Job, Prepared, Recommendation
from .queue import child_environment, claim, fail, recover
from .security import digest, endpoint, public_ip, seal, unseal
from .services import (
    approve,
    binding,
    enqueue,
    perform_study,
    write_artifact,
)
from .workloads import materialize, random_workload

TEST_KEY = Fernet.generate_key().decode()


@override_settings(INFIRO_CREDENTIAL_KEYS=TEST_KEY, ALLOWED_HOSTS=["testserver"])
class PlatformTests(TestCase):
    def setUp(self):
        self.root = TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.override = override_settings(INFIRO_PRIVATE_ROOT=Path(self.root.name))
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.alice = get_user_model().objects.create_user(
            "alice", password="a-secure-test-password"
        )
        self.bob = get_user_model().objects.create_user("bob", password="another-test-password")
        self.client.force_login(self.alice)

    def post(self, path, data):
        return self.client.post(
            "/app/api/" + path, json.dumps(data), content_type="application/json"
        )

    def connection(self, user=None, kind="huggingface", value="hf-private-secret", **metadata):
        c = Connection(
            owner=user or self.alice, name="Private " + kind, kind=kind, metadata=metadata
        )
        fields = {
            "huggingface": {"token": value},
            "nebius_inference": {"api_key": value},
            "nebius_jobs": {"iam_token": value},
            "storage": {"access_key_id": "key-id", "secret_access_key": value},
            "tavily": {"api_key": value},
        }
        c.ciphertext = seal(c, fields[kind])
        c.save()
        return c

    def test_preparation_only_worker_never_claims_or_recovers_studies(self):
        p = self.prepared()
        study = enqueue(self.alice, "study", p, {}, "leave-study-alone")
        preparation = enqueue(self.alice, "prepare", p, {}, "claim-prepare-only")
        claimed = claim({"prepare"})
        self.assertEqual(claimed.pk, preparation.pk)
        study.refresh_from_db()
        self.assertEqual(study.state, "queued")
        study.state = "running"
        study.lease_until = timezone.now() - timedelta(minutes=1)
        study.save()
        recover({"prepare"})
        study.refresh_from_db()
        self.assertEqual(study.state, "running")
        self.assertEqual(study.attempts, 0)

    def test_terminal_study_reports_partial_results_and_recovered_timeout(self):
        p = self.prepared()
        job = enqueue(self.alice, "study", p, {}, "partial-results")
        job.state = "completed"
        job.progress = {"provider_error": "Provider timed out", "provider_diagnostics": {
            "status": {"type": "TimeoutError"}}}
        job.result = {"summary": {"candidates": [{"framework": "sglang"}],
                                  "trial_state_counts": {
                                      "vllm": {"COMPLETE": 0, "FAIL": 1},
                                      "sglang": {"COMPLETE": 1, "FAIL": 0}}}}
        job.save()
        response = self.client.get("/app/api/jobs/" + str(job.id)).json()
        self.assertEqual(response["state"], "completed")
        self.assertEqual(response["outcome"], "partial_success")
        self.assertNotIn("provider_error", response["progress"])
        self.assertIn("recovered_provider_error", response["progress"])
        self.assertEqual(response["trial_failures"][0]["framework"], "vllm")
        job.result["summary"]["candidates"] = []
        job.save()
        response = self.client.get("/app/api/jobs/" + str(job.id)).json()
        self.assertEqual(response["outcome"], "no_valid_results")
        job.state = "queued"
        job.save()
        response = self.client.get("/app/api/jobs/" + str(job.id)).json()
        self.assertIn("provider_error", response["progress"])

    def test_study_logs_endpoint_is_owned_and_errors_are_safe(self):
        p = self.prepared()
        job = enqueue(self.alice, "study", p, {}, "log-endpoint")
        job.controller_run = "existing-controller"
        job.save()
        url = "/app/api/jobs/" + str(job.id) + "/diagnostics"
        payload = {"study_id": str(job.id), "frameworks": []}
        with patch("infiro_platform.study_logs.saved_logs", return_value=payload) as read:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), payload)
            self.assertIn("no-store", response["Cache-Control"])
            self.client.force_login(self.bob)
            self.assertEqual(self.client.get(url).status_code, 404)
            self.assertEqual(read.call_count, 1)
        self.client.force_login(self.alice)
        with patch("infiro_platform.study_logs.saved_logs",
                   side_effect=RuntimeError("private-token")):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private-token", response.content.decode())
        self.client.logout()
        self.assertEqual(self.client.get(url).status_code, 401)

    def test_saved_log_scrubbing_and_incomplete_request_diagnostics(self):
        from .study_logs import log_reasons, scrub_log

        raw = ("Bearer private-token https://s3.example/log?signature=hidden "
               "WANDB_API_KEY=hidden-key\nSuccessful requests: 99\n"
               "Optuna trial 0 failed for config: Benchmark is failed or incomplete; "
               "objective is invalid")
        cleaned = scrub_log(raw, ["private-token"])
        self.assertNotIn("private-token", cleaned)
        self.assertNotIn("hidden", cleaned)
        reasons = log_reasons(cleaned, 100)
        self.assertIn("Trial 0: Benchmark is failed or incomplete", reasons[0])
        self.assertIn("99/100 requests succeeded", reasons[1])

    def test_two_trials_survive_api_preparation_and_approval(self):
        from .services import perform_prepare
        from .workloads import random_workload

        p = self.prepared()
        c = self.connection(kind="nebius_inference", endpoint="https://example.com/v1")
        choices = Choices(trials=2, num_prompts=3, input_tokens=16, output_tokens=8).model_dump()
        response = self.post("prepared", {"recommendation_id": str(p.recommendation_id),
            "choices": choices, "connection_refs": {"nebius_inference": str(c.id)},
            "submission_key": "two-trials"})
        self.assertEqual(response.status_code, 202)
        prepared = Prepared.objects.get(pk=response.json()["prepared"]["id"])
        self.assertEqual(prepared.choices["trials"], 2)
        preparation = Job.objects.get(pk=response.json()["job"]["id"])
        with (
            patch("hf_model_watch.optimization.preparation.research",
                  return_value={"sources": [], "inspection": {"heads": {}, "claims": []}}),
            patch("infiro_platform.workloads.random_workload", side_effect=lambda config:
                  random_workload(config, tokenizer=FixtureTokenizer())),
        ):
            perform_prepare(preparation)
        prepared.refresh_from_db()
        self.assertEqual(prepared.payload["optuna"]["trials"], 2)
        self.assertEqual(prepared.summary["optuna"]["trials"], 2)
        job = approve(self.alice, prepared.id, prepared.digest, True)
        self.assertEqual(job.target_id, prepared.id)
        from llm_optimizer.remote.worker import optimizer_command
        args = optimizer_command({"request": prepared.payload, "framework": "vllm"},
                                 Path(self.root.name) / "trial-command/results.json")
        self.assertEqual(args[args.index("--optuna-trials") + 1], "2")

    def test_wandb_reference_is_pinned_before_approval_hash(self):
        from .services import finalize_preparation

        p = seed_prepared(self.alice)
        p.choices["dry_run"] = False
        p.payload["optuna"] = {"trials": p.choices["trials"], "objectives": p.choices["objectives"]}
        p.payload["images"] = {}
        tracking = {"project": "test", "entity": "team", "secret_version_id": "parent-secret"}
        p.summary = {"tracking": tracking, "workload_identity": {}}
        p.summary["draft_sha256"] = digest({"payload": p.payload, "intent": p.intent,
                                            "workload_identity": {}})
        j = enqueue(self.alice, "prepare", p, {}, "wandb-pin")
        pinned = tracking | {"secret_version_id": "immutable-version"}
        with (
            patch("infiro_platform.nebius_provider.validate_live_configuration"),
            patch("infiro_platform.nebius_provider.pin_secret_references",
                  return_value={"wandb": pinned}) as resolve,
        ):
            finalize_preparation(j, p, {})
        self.assertEqual(resolve.call_args.args[0]["wandb"].metadata, tracking)
        self.assertEqual(p.summary["tracking"], pinned)
        self.assertEqual(p.digest, digest(binding(p)))

    def test_recommendation_names_creation_validation_and_fallback(self):
        c = self.connection(kind="nebius_inference", endpoint="https://example.com/v1")
        data = {"requirements": {"request": "A small coding assistant"},
                "connection_refs": {"nebius_inference": str(c.id)}}
        for bad in [None, 123, "x" * 161, "a\nb", "a\u202eb"]:
            self.assertEqual(self.post("recommendations", {**data, "name": bad}).status_code, 400)
        for name in ["  Coding research  ", "   "]:
            response = self.post("recommendations", {**data, "name": name, "submission_key": name})
            self.assertEqual(response.status_code, 202)
            rec = Recommendation.objects.get(pk=response.json()["recommendation"]["id"])
            url = "/app/api/recommendations/" + str(rec.id)
            display = self.client.get(url).json()["name"]
            if name.strip():
                self.assertEqual(display, "Coding research")
            else:
                self.assertIn("A small coding assistant", display)
                self.assertIn(rec.id.hex[:8], display)
                self.assertIn(rec.created_at.strftime("%Y-%m-%d %H:%M"), display)
                self.assertEqual(rec.name, "")
            self.assertEqual(display, self.client.get(url).json()["name"])
            replay = self.post(
                "recommendations", {**data, "name": "Different", "submission_key": name}
            )
            self.assertEqual(replay.json()["recommendation"]["name"], display)

    def test_recommendation_rename_preserves_all_execution_metadata(self):
        p = self.prepared()
        rec = p.recommendation
        job = enqueue(self.alice, "prepare", p, {}, "rename-metadata")
        artifact = write_artifact(job, "request.json", p.payload)
        before = Prepared.objects.get(pk=p.pk).__dict__.copy()
        before.pop("_state")
        jobs_before = list(Job.objects.values())
        url = "/app/api/recommendations/" + str(rec.id)
        def rename(payload):
            return self.client.patch(url, json.dumps(payload), content_type="application/json")
        for bad in [None, "x" * 161, "a\nb"]:
            self.assertEqual(rename({"name": bad}).status_code, 400)
        self.assertEqual(rename({"name": "Safe", "model_id": "changed"}).status_code, 400)
        response = rename({"name": "  <script>alert(1)</script>  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "<script>alert(1)</script>")
        old_version, old_selected = rec.version, rec.selected
        rec.refresh_from_db()
        self.assertEqual((rec.version, rec.selected), (old_version, old_selected))
        after = Prepared.objects.get(pk=p.pk).__dict__.copy()
        after.pop("_state")
        self.assertEqual(before, after)
        self.assertEqual(jobs_before, list(Job.objects.values()))
        self.assertEqual(artifact.sha256, Artifact.objects.get(pk=artifact.pk).sha256)
        self.assertEqual(
            self.client.get("/app/api/prepared/" + str(p.id)).json()["recommendation_name"],
            rec.name,
        )
        self.client.force_login(self.bob)
        self.assertEqual(rename({"name": "Stolen"}).status_code, 404)
        self.client.logout()
        self.assertEqual(rename({"name": "Anonymous"}).status_code, 401)

    def test_study_name_creation_and_validation(self):
        p = self.prepared()
        c = self.connection(kind="nebius_inference", endpoint="https://example.com/v1")
        data = {"recommendation_id": str(p.recommendation_id),
                "connection_refs": {"nebius_inference": str(c.id)}}
        for bad in [None, 123, "x" * 161, "a\nb", "a\u202eb"]:
            self.assertEqual(self.post("prepared", {**data, "name": bad}).status_code, 400)
        for name in ["  Comparison A  ", "   "]:
            response = self.post("prepared", {**data, "name": name, "submission_key": name})
            self.assertEqual(response.status_code, 202)
            saved = Prepared.objects.get(pk=response.json()["prepared"]["id"])
            if name.strip():
                self.assertEqual(saved.name, "Comparison A")
            else:
                self.assertIn("model · ", saved.name)
                self.assertIn(saved.id.hex[:8], saved.name)
            replay = self.post("prepared", {**data, "name": "Different", "submission_key": name})
            self.assertEqual(replay.json()["prepared"]["name"], saved.name)
            self.assertNotIn("name", saved.choices)

    def test_legacy_study_names_are_stable_without_writes(self):
        p = self.prepared()
        url = "/app/api/prepared/" + str(p.id)
        first = self.client.get(url).json()
        self.assertIn("model · " + p.created_at.strftime("%Y-%m-%d %H:%M"), first["name"])
        self.assertIn(p.id.hex[:8], first["name"])
        self.assertEqual(first["model_id"], "org/model")
        self.assertEqual(first["name"], self.client.get(url).json()["name"])
        p.refresh_from_db()
        self.assertEqual(p.name, "")
        self.assertEqual(p.digest, digest(binding(p)))

    def test_rename_preserves_approval_and_execution_identity_and_ownership(self):
        p = self.prepared()
        before = binding(p)
        original_digest = p.digest
        url = "/app/api/prepared/" + str(p.id)
        def rename(name):
            return self.client.patch(
                url, json.dumps({"name": name}), content_type="application/json"
            )
        self.assertEqual(rename("  Before approval  ").status_code, 200)
        job = approve(self.alice, p.id, original_digest, True)
        artifact = write_artifact(job, "request.json", p.payload)
        job.controller_run = "existing-controller"
        job.progress = {"jobs": [{"id": "provider-job"}], "wandb_run_id": "existing-wandb"}
        job.result = {"measurements": [1, 2]}
        job.save()
        for state in ["queued", "completed", "cancelled"]:
            Job.objects.filter(pk=job.pk).update(state=state)
            job.refresh_from_db()
            execution = Job.objects.filter(pk=job.pk).values().get()
            prepared_before = Prepared.objects.filter(pk=p.pk).values().get()
            self.assertEqual(rename("  <script>alert(1)</script>  ").status_code, 200)
            p.refresh_from_db()
            self.assertEqual(p.name, "<script>alert(1)</script>")
            self.assertEqual(binding(p), before)
            self.assertEqual(p.digest, original_digest)
            prepared_after = Prepared.objects.filter(pk=p.pk).values().get()
            prepared_before.pop("name")
            prepared_after.pop("name")
            self.assertEqual(prepared_before, prepared_after)
            self.assertEqual(Job.objects.filter(pk=job.pk).values().get(), execution)
            self.assertEqual(Artifact.objects.get(pk=artifact.pk).sha256, artifact.sha256)
        self.assertEqual(approve(self.alice, p.id, original_digest, True).id, job.id)
        self.client.force_login(self.bob)
        self.assertEqual(rename("Stolen").status_code, 404)
        self.client.logout()
        self.assertEqual(rename("Anonymous").status_code, 401)

    def test_configuration_downloads_remain_available_after_preparation_and_study(self):
        prepared = self.prepared()
        preparation = enqueue(self.alice, "prepare", prepared, {}, "download-preparation")
        artifact = write_artifact(preparation, "request.json", {"model": "org/model"})
        preparation.state = "completed"
        preparation.save()
        enqueue(self.alice, "study", prepared, {}, "download-study")
        response = self.client.get("/app/api/prepared/" + str(prepared.id))
        self.assertEqual(response.status_code, 200)
        files = response.json()["configuration_files"]
        self.assertEqual([f["name"] for f in files], ["request.json"])
        self.assertEqual(self.client.get(files[0]["download_url"]).status_code, 200)
        self.client.force_login(self.bob)
        self.assertEqual(self.client.get(files[0]["download_url"]).status_code, 404)
        self.assertEqual(
            self.client.get("/app/api/prepared/" + str(prepared.id)).status_code, 404
        )
        self.assertEqual(artifact.job_id, preparation.id)

    def test_study_logs_uses_latest_study_and_existing_optimizer_log_reader(self):
        import io

        from django.core.management import call_command

        prepared = self.prepared()
        prepared.payload = {"bucket": "private-bucket", "prefix": "private-study"}
        prepared.save()
        storage = self.connection(kind="storage", region="eu-north1", bucket="private-bucket")
        job = enqueue(self.alice, "study", prepared, {"storage": str(storage.id)}, "log-study")
        job.controller_run = "controller-run"
        job.save()
        output = io.StringIO()
        module = "infiro_platform.management.commands.study_logs"
        with (
            patch(module + ".provider_jobs", return_value={"vllm": "aijob-latest"}),
            patch(module + ".boto3.client") as client,
            patch(module + ".S3Artifacts") as artifacts,
        ):
            artifacts.return_value.execution_log_tail.return_value = "saved execution output"
            call_command("study_logs", latest=True, stdout=output)
        self.assertIn("aijob-latest", output.getvalue())
        self.assertIn("saved execution output", output.getvalue())
        self.assertNotIn("hf-private-secret", output.getvalue())
        artifacts.return_value.execution_log_tail.assert_called_with("controller-run", "vllm")
        client.return_value.close.assert_called_once()

    def prepared(self):
        rec = Recommendation.objects.create(
            owner=self.alice,
            requirements={"request": "test"},
            selected={"model_id": "org/model", "revision": "a" * 40},
        )
        p = Prepared.objects.create(
            owner=self.alice,
            recommendation=rec,
            recommendation_version=rec.version,
            choices=Choices().model_dump(),
            status="prepared",
            payload={"model": "org/model"},
            intent={},
        )
        p.digest = digest(binding(p))
        p.save()
        return p

    def test_service_account_connection_encryption_and_child_isolation(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        metadata = {
            "project_id": "project-test",
            "auth_method": "service_account",
            "service_account_id": "serviceaccount-test",
            "public_key_id": "publickey-test",
        }
        response = self.post(
            "connections",
            {
                "name": "Service account",
                "kind": "nebius_jobs",
                "metadata": metadata,
                "credentials": {"private_key": pem},
            },
        )
        self.assertEqual(response.status_code, 201, response.content)
        connection = Connection.objects.get(name="Service account")
        self.assertNotIn(pem, connection.ciphertext)
        self.assertEqual(unseal(connection), {"private_key": pem})
        bootstrap = self.client.get("/app/api/bootstrap").json()
        self.assertNotIn(pem, json.dumps(bootstrap))
        self.assertEqual(bootstrap["connections"][0]["credentials"], {"private_key": "••••••••"})
        job = enqueue(
            self.alice,
            "validate",
            connection,
            {"nebius_jobs": str(connection.id)},
            "sa-validation",
        )
        env = child_environment(job)
        self.assertEqual(env["NEBIUS_PRIVATE_KEY"], pem)
        self.assertNotIn("NEBIUS_IAM_TOKEN", env)
        self.assertNotIn(pem, json.dumps(job.connection_refs))
        self.client.force_login(self.bob)
        self.assertEqual(self.client.get("/app/api/bootstrap").json()["connections"], [])
        self.assertEqual(self.post("connections/" + str(connection.id), {}).status_code, 404)

    def test_service_account_validation_and_sdk_credentials(self):
        import base64

        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from nebius.base.service_account.service_account import ServiceAccount

        from .nebius_auth import make_sdk
        from .security import validate_connection

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        metadata = {
            "project_id": "project-test",
            "auth_method": "service_account",
            "service_account_id": "serviceaccount-test",
            "public_key_id": "publickey-test",
        }
        values = {"private_key": base64.b64encode(pem).decode()}
        self.assertEqual(validate_connection("nebius_jobs", metadata, values), metadata)
        with patch("nebius.sdk.SDK") as sdk:
            make_sdk(metadata, values)
            credentials = sdk.call_args.kwargs["credentials"]
            self.assertIsInstance(credentials, ServiceAccount)
            self.assertEqual(credentials.service_account_id, metadata["service_account_id"])
            self.assertEqual(credentials.public_key_id, metadata["public_key_id"])
            self.assertEqual(credentials.private_key.private_numbers(), key.private_numbers())
        for invalid in (
            {"private_key": "invalid-secret"},
            {"iam_token": "wrong-mode"},
            {"private_key": values["private_key"], "iam_token": "mixed"},
        ):
            with self.assertRaises(ValueError) as error:
                validate_connection("nebius_jobs", metadata, invalid)
            self.assertNotIn("invalid-secret", str(error.exception))
        with self.assertRaises(ValueError):
            validate_connection(
                "nebius_jobs",
                {"project_id": "project-test", "auth_method": "service_account"},
                values,
            )

    def test_secretstash_connection_ids(self):
        for version_id in ("mbse-test123", "mbseTest123", "mbsec-secret123", "mbsecver-legacy123"):
            response = self.post(
                "connections",
                {
                    "kind": "huggingface",
                    "name": version_id,
                    "metadata": {"secret_version_id": version_id},
                    "credentials": {"token": "hf-test"},
                },
            )
            self.assertEqual(response.status_code, 201, response.content)
            self.assertEqual(
                Connection.objects.get(name=version_id).metadata["secret_version_id"], version_id
            )
        for invalid in ("mbse-", "mbse-../private", "not-an-id"):
            response = self.post(
                "connections",
                {
                    "kind": "huggingface",
                    "name": "invalid",
                    "metadata": {"secret_version_id": invalid},
                    "credentials": {"token": "hf-test"},
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("SecretStash", response.content.decode())

    def test_failure_diagnostics_never_include_exception_values(self):
        from .diagnostics import failure_details, failure_summary

        try:
            try:
                raise MemoryError("private-prompt hf-secret https://host/?token=secret")
            except MemoryError as exc:
                raise ValueError("another-secret") from exc
        except ValueError as exc:
            details = failure_details(exc)
        serialized = json.dumps(details)
        for secret in ("private-prompt", "hf-secret", "https://", "another-secret"):
            self.assertNotIn(secret, serialized)
        self.assertEqual([e["type"] for e in details["exceptions"]], ["ValueError", "MemoryError"])
        self.assertIn("memory limit", failure_summary(details))
        from grpc import StatusCode
        from nebius.aio.service_error import RequestError

        exception = RequestError(
            SimpleNamespace(
                code=StatusCode.INVALID_ARGUMENT,
                service_errors=[
                    SimpleNamespace(code="JwtKeyNotExists", service="token-service", details=None)
                ],
                message="private-message",
            )
        )
        details = failure_details(exception)
        self.assertNotIn("private-message", json.dumps(details))
        self.assertIn("registered public-key ID", failure_summary(details))
        self.assertIn(
            "Nebius inference authentication failed",
            failure_summary(
                {
                    "exceptions": [
                        {"type": "AuthenticationError", "module": "openai", "http_status": 401}
                    ]
                }
            ),
        )

    def test_authentication_and_csrf(self):
        anonymous = Client(enforce_csrf_checks=True)
        self.assertEqual(anonymous.get("/app").status_code, 302)
        self.assertEqual(anonymous.get("/app/").status_code, 302)
        self.assertEqual(anonymous.get("/app/api/bootstrap").status_code, 401)
        self.assertEqual(
            anonymous.post("/app/login", {"username": "alice", "password": "bad"}).status_code, 403
        )
        anonymous.get("/app/login")
        csrf = anonymous.cookies["csrftoken"].value
        response = anonymous.post(
            "/app/login",
            {
                "username": "alice",
                "password": "a-secure-test-password",
                "csrfmiddlewaretoken": csrf,
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(anonymous.get("/app/api/bootstrap").status_code, 200)
        self.assertEqual(
            anonymous.post(
                "/app/api/connections", "{}", content_type="application/json"
            ).status_code,
            403,
        )
        response = anonymous.get("/app/api/bootstrap")
        self.assertEqual(response["Cache-Control"], "private, no-store")

    def test_registration_and_invalid_login_are_real_sessions(self):
        client = Client(enforce_csrf_checks=True)
        client.get("/app/register")
        response = client.post(
            "/app/register",
            {
                "username": "new-user",
                "password": "a-new-long-unique-password",
                "csrfmiddlewaretoken": client.cookies["csrftoken"].value,
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(client.get("/app/api/bootstrap").json()["user"]["username"], "new-user")

    def test_credential_encryption_masking_and_envelope_ownership(self):
        c = self.connection()
        self.assertNotIn("hf-private-secret", c.ciphertext)
        self.assertEqual(unseal(c), {"token": "hf-private-secret"})
        raw = self.client.get("/app/api/bootstrap").content.decode()
        self.assertNotIn("hf-private-secret", raw)
        self.assertNotIn(c.ciphertext, raw)
        c.owner = self.bob
        with self.assertRaises(ValueError):
            unseal(c)

    @override_settings(INFIRO_CREDENTIAL_KEYS="")
    def test_credentials_require_external_persistent_key(self):
        response = self.post(
            "connections",
            {"kind": "huggingface", "name": "Token", "credentials": {"token": "secret"}},
        )
        self.assertEqual(response.status_code, 503)
        self.assertFalse(Connection.objects.exists())

    def test_connection_replacement_deletion_and_stale_versions(self):
        c = self.connection()
        p = self.prepared()
        p.connection_refs = {"huggingface": str(c.id)}
        p.connection_versions = {"huggingface": 1}
        p.digest = digest(binding(p))
        p.save()
        result = self.client.put(
            "/app/api/connections/" + str(c.id),
            json.dumps({"credentials": {"token": "replacement"}}),
            content_type="application/json",
        )
        self.assertEqual(result.status_code, 200)
        self.assertNotIn("replacement", result.content.decode())
        c.refresh_from_db()
        self.assertEqual(c.version, 2)
        self.assertEqual(unseal(c)["token"], "replacement")
        p.refresh_from_db()
        self.assertTrue(p.invalidated)
        self.assertEqual(self.client.delete("/app/api/connections/" + str(c.id)).status_code, 200)

    def test_two_user_isolation_for_every_resource_and_mutation(self):
        c = self.connection(self.bob)
        rec = Recommendation.objects.create(
            owner=self.bob, requirements={"request": "Bob private prompt"}, status="recommended"
        )
        d = Dataset.objects.create(
            owner=self.bob,
            name="Bob dataset",
            config={"repo_id": "private/data"},
            preview=[{"prompt": "Bob private prompt"}],
        )
        p = Prepared.objects.create(
            owner=self.bob, recommendation=rec, dataset=d, recommendation_version=1, choices={}
        )
        j = Job.objects.create(
            owner=self.bob,
            kind="study",
            target_id=p.id,
            available_at=timezone.now(),
            dedupe_key="bob-job",
            result={"private": "Bob results"},
        )
        a = write_artifact(j, "private.json", {"prompt": "Bob private prompt"})
        for route, obj in [
            ("connections", c),
            ("recommendations", rec),
            ("datasets", d),
            ("prepared", p),
            ("jobs", j),
            ("artifacts", a),
        ]:
            self.assertEqual(self.client.get(f"/app/api/{route}/{obj.id}").status_code, 404, route)
            if route not in {"datasets", "artifacts"}:
                self.assertEqual(
                    self.post(
                        f"{route}/{obj.id}", {"approved": True, "model_id": "org/model"}
                    ).status_code,
                    404,
                    route,
                )
        response = self.client.get("/app/api/bootstrap")
        self.assertNotIn("Bob", response.content.decode())
        self.assertEqual(
            self.post(
                "datasets",
                {
                    "config": {"repo_id": "org/data", "model": "org/model",
                               "data_file": "data.json"},
                    "connection_refs": {"huggingface": str(c.id)},
                },
            ).status_code,
            404,
        )
        self.assertEqual(self.post("prepared", {"recommendation_id": str(rec.id)}).status_code, 404)
        mine = self.prepared()
        self.assertEqual(
            self.post(
                "prepared",
                {"recommendation_id": str(mine.recommendation_id), "dataset_id": str(d.id)},
            ).status_code,
            404,
        )

    def test_per_job_credential_and_cache_isolation_without_environment_mutation(self):
        a = self.connection()
        b = self.connection(self.bob, value="hf-bob-secret")
        inference = self.connection(
            kind="nebius_inference", value="research-key", endpoint="https://example.com/v1"
        )
        jobs = []
        for user, c in [(self.alice, a), (self.bob, b)]:
            jobs.append(enqueue(user, "dataset", c, {"huggingface": str(c.id)}, "prepare-data"))
        before = os.environ.copy()
        with patch.dict(
            os.environ, {"HF_TOKEN": "ambient-token", "AWS_SECRET_ACCESS_KEY": "ambient-aws"}
        ):
            envs = [child_environment(j) for j in jobs]
            self.assertEqual(os.environ["HF_TOKEN"], "ambient-token")
        self.assertEqual(os.environ, before)
        self.assertEqual(envs[0]["HF_TOKEN"], "hf-private-secret")
        self.assertEqual(envs[1]["HF_TOKEN"], "hf-bob-secret")
        self.assertNotEqual(envs[0]["HF_HOME"], envs[1]["HF_HOME"])
        for env in envs:
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
            self.assertNotIn("NEBIUS_API_KEY", env)
            self.assertNotIn("INFIRO_CREDENTIAL_KEYS", env)
            self.assertNotIn("DJANGO_SECRET_KEY", env)
        study = enqueue(
            self.alice, "study", a, {"nebius_inference": str(inference.id)}, "study-env"
        )
        self.assertNotIn("NEBIUS_API_KEY", child_environment(study))
        self.assertNotIn("hf-private-secret", json.dumps(jobs[0].connection_refs))

    def test_exact_approval_duplicate_protection_and_mutated_payload(self):
        p = self.prepared()
        with self.assertRaises(ValueError):
            approve(self.alice, p.id, p.digest, False)
        with self.assertRaises(ValueError):
            approve(self.alice, p.id, "wrong", True)
        one = approve(self.alice, p.id, p.digest, True)
        two = approve(self.alice, p.id, p.digest, True)
        self.assertEqual(one.id, two.id)
        self.assertEqual(Job.objects.filter(kind="study").count(), 1)
        p.payload["gpu_count"] = 8
        p.save()
        with self.assertRaises(ValueError):
            approve(self.alice, p.id, p.digest, True)

    def test_stale_selection_and_connection_approval(self):
        p = self.prepared()
        p.recommendation.version += 1
        p.recommendation.save()
        with self.assertRaises(ValueError):
            approve(self.alice, p.id, p.digest, True)
        p = self.prepared()
        c = self.connection()
        p.connection_refs = {"huggingface": str(c.id)}
        p.connection_versions = {"huggingface": 1}
        p.digest = digest(binding(p))
        p.save()
        Connection.objects.filter(pk=c.pk).update(version=2)
        with self.assertRaises(ValueError):
            approve(self.alice, p.id, p.digest, True)

    def test_worker_recovery_bounded_retries_and_claim(self):
        p = self.prepared()
        j = enqueue(self.alice, "prepare", p, {}, "recover")
        self.assertEqual(claim().id, j.id)
        self.assertIsNone(claim())
        Job.objects.filter(pk=j.id).update(lease_until=timezone.now() - timedelta(seconds=1))
        recover()
        j.refresh_from_db()
        self.assertEqual(j.state, "queued")
        self.assertEqual(j.attempts, 1)
        Job.objects.filter(pk=j.id).update(
            state="running", attempts=3, lease_until=timezone.now() - timedelta(seconds=1)
        )
        recover()
        j.refresh_from_db()
        self.assertEqual(j.state, "failed")
        self.assertEqual(Prepared.objects.get(pk=p.id).status, "failed")

        rejected = self.prepared()
        deterministic = enqueue(self.alice, "prepare", rejected, {}, "oversized-source")
        deterministic.attempts = 1
        fail(
            deterministic,
            "Selected dataset files exceed the 1 GiB download limit, or sizes are unknown.",
        )
        deterministic.refresh_from_db()
        self.assertEqual(deterministic.state, "failed")
        self.assertEqual(
            deterministic.error,
            "Selected dataset files exceed the 1 GiB download limit, or sizes are unknown.",
        )
        self.assertEqual(Prepared.objects.get(pk=rejected.id).status, "failed")

    def test_cancel_before_submission_and_active_connection_protection(self):
        p = self.prepared()
        j = approve(self.alice, p.id, p.digest, True)
        self.post("jobs/" + str(j.id), {})
        j.refresh_from_db()
        perform_study(j)
        self.assertEqual(j.state, "cancelled")
        self.assertFalse(j.controller_run)
        c = self.connection()
        enqueue(self.alice, "dataset", c, {"huggingface": str(c.id)}, "using-token")
        self.assertEqual(self.client.delete("/app/api/connections/" + str(c.id)).status_code, 400)

    def test_redaction_of_provider_secrets_and_artifact_integrity(self):
        c = self.connection()
        j = enqueue(self.alice, "dataset", c, {}, "artifact")
        with patch.dict(os.environ, {"HF_TOKEN": "very-sensitive-token"}):
            a = write_artifact(
                j,
                "results.json",
                {"message": "very-sensitive-token", "api_key": "x", "secret_access_key": "y"},
            )
        response = self.client.get("/app/api/artifacts/" + str(a.id))
        raw = b"".join(response.streaming_content).decode()
        self.assertNotIn("very-sensitive-token", raw)
        self.assertNotIn("api_key", raw)
        (
            Path(self.root.name)
            / "users"
            / str(self.alice.id)
            / "jobs"
            / str(j.id)
            / "results.json"
        ).write_text("tampered")
        self.assertEqual(self.client.get("/app/api/artifacts/" + str(a.id)).status_code, 409)

    def test_ssrf_private_addresses_and_dns_answers(self):
        for value in [
            "127.0.0.1",
            "10.0.0.1",
            "169.254.169.254",
            "::1",
            "::ffff:127.0.0.1",
            "192.168.0.1",
            "0.0.0.0",
            "100.64.0.1",
        ]:
            with self.assertRaises(ValueError):
                public_ip(value)
        for value in [
            "http://example.com",
            "https://user:secret@example.com",
            "https://example.com/?token=secret",
            "https://example.com:8000",
            "file:///etc/passwd",
        ]:
            with self.assertRaises(ValueError):
                endpoint(value)
        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
        ):
            with self.assertRaises(ValueError):
                endpoint("https://public-looking.example")

    def test_network_guard_catches_redirect_peer_and_rebinding(self):
        import ssl

        from .security import install_network_guard

        with (
            patch.object(socket.socket, "connect", Mock()) as connect,
            patch.object(socket.socket, "connect_ex", Mock()),
            patch.object(socket.socket, "recv"),
            patch.object(socket.socket, "recv_into"),
            patch.object(ssl.SSLSocket, "read"),
        ):
            install_network_guard()
            sock = SimpleNamespace(family=socket.AF_INET)
            with self.assertRaises(ValueError):
                socket.socket.connect(sock, ("127.0.0.1", 443))
            with patch(
                "socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))],
            ):
                with self.assertRaises(ValueError):
                    socket.socket.connect(sock, ("rebound.example", 443))
            connect.assert_not_called()

    def test_dataset_repeatability_failure_and_budget_validation(self):
        config, a = fixture_manifest()
        _, b = fixture_manifest()
        self.assertEqual(a, b)
        self.assertEqual(len(set(a["row_indices"])), 3)
        api = SimpleNamespace(
            dataset_info=lambda *a, **k: SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(size=1024, rfilename="data.json")]
            ),
            model_info=lambda *a, **k: SimpleNamespace(sha="a" * 40, siblings=[]),
        )
        for rows in [[{}], [{"conversations": [{"from": "gpt", "value": "x"}]}], []]:
            with self.assertRaises(ValueError):
                materialize(
                    config,
                    api=api,
                    loader=lambda *a, rows=rows, **kw: rows,
                    tokenizer=FixtureTokenizer(),
                )
        with self.assertRaisesRegex(ValueError, "reduce prompt count or choose another split"):
            materialize(config, api=api, loader=lambda *a, **kw: [], tokenizer=FixtureTokenizer())
        api.dataset_info = lambda *a, **kw: SimpleNamespace(
            sha="a" * 40, siblings=[SimpleNamespace(size=2 * 1024**3, rfilename="data.json")]
        )
        with self.assertRaisesRegex(ValueError, "1 GiB"):
            materialize(config, api=api, loader=Mock(), tokenizer=FixtureTokenizer())
        with self.assertRaises(ValueError):
            Choices(num_prompts=1000, input_tokens=16384)
        with self.assertRaises(ValueError):
            Choices(gpu_count=8, gpu_hourly_cost_usd=10, budget_usd_per_hour=5)

    def test_custom_dataset_validation_progress_pinning_and_frozen_reuse(self):
        from .services import perform_dataset, perform_prepare, read_workload
        from .workloads import identity

        config, manifest = fixture_manifest()
        dataset = Dataset.objects.create(owner=self.alice, name="Custom source", config=config)
        job = enqueue(self.alice, "dataset", dataset, {}, "custom-freeze")
        stages = []
        def freeze(config, *, progress):
            progress("Validating ShareGPT conversations and freezing the first human turns")
            stages.append(Job.objects.get(pk=job.pk).progress["stage"])
            return manifest
        with patch("infiro_platform.workloads.materialize", side_effect=freeze) as sampler:
            perform_dataset(job)
            perform_dataset(job)
        sampler.assert_called_once()
        self.assertTrue(stages)
        dataset.refresh_from_db()
        self.assertEqual(dataset.status, "ready")
        self.assertEqual(dataset.identity, identity(manifest))
        self.assertLessEqual(len(dataset.preview), 3)
        self.assertTrue(all(len(row["prompt"]) <= 400 for row in dataset.preview))
        self.assertEqual(read_workload(dataset), manifest)

        base = seed_prepared(self.alice)
        c = self.connection(kind="nebius_inference", endpoint="https://example.com/v1")
        choices = base.choices | {
            "workload_mode": "sharegpt", "frameworks": ["vllm", "sglang"], "trials": 7
        }
        data = {"recommendation_id": str(base.recommendation_id), "dataset_id": str(dataset.id),
                "choices": choices, "connection_refs": {"nebius_inference": str(c.id)}}
        # Ownership and readiness are enforced before queuing any preparation.
        self.client.force_login(self.bob)
        self.assertEqual(self.post("prepared", data).status_code, 404)
        self.client.force_login(self.alice)
        Dataset.objects.filter(pk=dataset.pk).update(status="validating")
        self.assertEqual(self.post("prepared", data).status_code, 400)
        Dataset.objects.filter(pk=dataset.pk).update(status="ready")
        response = self.post("prepared", data)
        self.assertEqual(response.status_code, 202)
        prepared = Prepared.objects.get(pk=response.json()["prepared"]["id"])
        preparation = Job.objects.get(pk=response.json()["job"]["id"])
        with (
            patch("hf_model_watch.optimization.preparation.research",
                  return_value={"sources": [], "inspection": {"claims": [], "heads": []}}),
            patch("infiro_platform.workloads.materialize") as sampler,
            patch("infiro_platform.workloads.random_workload") as random_sampler,
        ):
            perform_prepare(preparation)
        sampler.assert_not_called()
        random_sampler.assert_not_called()
        prepared.refresh_from_db()
        self.assertEqual(prepared.payload["workload_manifest"], manifest)
        self.assertEqual(prepared.summary["workload_identity"], dataset.identity)
        self.assertEqual(prepared.payload["frameworks"], ["vllm", "sglang"])
        self.assertEqual(prepared.payload["optuna"]["trials"], 7)
        self.assertEqual(prepared.summary["workload_preview"], dataset.preview)

    def test_discovery_completion_cannot_overwrite_a_concurrent_rename(self):
        from .services import perform_discovery
        rec = Recommendation.objects.create(
            owner=self.alice, requirements={"request": "Coding model"}
        )
        job = enqueue(self.alice, "discover", rec, {}, "rename-during-discovery")
        def discover(*args):
            Recommendation.objects.filter(pk=rec.pk).update(name="Renamed during research")
            return {"recommendations": []}
        with patch("hf_model_watch.recommender.discovery.discover_models", side_effect=discover):
            perform_discovery(job)
        rec.refresh_from_db()
        self.assertEqual(rec.name, "Renamed during research")
        self.assertEqual(rec.status, "empty")

    def test_large_sharegpt_file_is_preflighted_independently_and_pinned(self):
        config, _ = fixture_manifest()
        config["data_file"] = "conversations.json"
        large_file = SimpleNamespace(size=800 * 1024**2, rfilename="conversations.json")
        unrelated = SimpleNamespace(size=4 * 1024**3, rfilename="archive/old-data.json")
        api = SimpleNamespace(
            dataset_info=lambda *a, **k: SimpleNamespace(
                sha="b" * 40, siblings=[large_file, unrelated]
            ),
            model_info=lambda *a, **k: SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(size=1024, rfilename="tokenizer.json")]
            ),
        )
        rows = [
            {"conversations": [{"from": "human", "value": f"Prompt {i}"}]}
            for i in range(5)
        ]
        payload = json.dumps(rows, indent=2).encode()
        with patch("huggingface_hub.HfFileSystem") as filesystem:
            filesystem.return_value.open.return_value = io.BytesIO(payload)
            manifest = materialize(config, api=api, tokenizer=FixtureTokenizer())
        self.assertEqual(manifest["dataset"]["data_file"], "conversations.json")
        self.assertEqual(manifest["dataset"]["revision"], "b" * 40)
        filesystem.return_value.open.assert_called_once_with(
            f"datasets/{config['repo_id']}@{'b' * 40}/conversations.json",
            "rb", block_size=1024 * 1024, cache_type="readahead",
        )
        expected = materialize(
            config, api=api, loader=lambda *a, **kw: rows, tokenizer=FixtureTokenizer()
        )
        self.assertEqual(manifest, expected)

    def test_incremental_sharegpt_parser_memory_and_frozen_identity(self):
        import tracemalloc

        from .workloads import MAX_ROW_BYTES, WorkloadLimitError, iter_json_records

        # A virtual source much larger than parser memory; no full test payload allocation.
        class RepeatedStream:
            def __init__(self, row, count):
                self.row, self.count, self.pending = row, count, b"["
                self.read_sizes = []

            def read(self, size):
                self.assert_size(size)
                while len(self.pending) < size and self.count:
                    self.pending += self.row + (b"," if self.count > 1 else b"]")
                    self.count -= 1
                chunk, self.pending = self.pending[:size], self.pending[size:]
                return chunk

            def assert_size(self, size):
                if not 0 < size <= 64 * 1024:
                    raise AssertionError("unbounded source read")
                self.read_sizes.append(size)

        rows = [{"conversations": [
            {"from": "system", "value": "system"},
            {"from": "human", "value": 'First é prompt } [ \" ' + str(i)},
            {"from": "gpt", "value": "answer"},
            {"from": "human", "value": "Never sampled"},
        ]} for i in range(12)]
        encoded = json.dumps(rows[0], ensure_ascii=False).encode()
        stream = RepeatedStream(encoded, 20000)
        tracemalloc.start()
        try:
            self.assertEqual(sum(1 for _ in iter_json_records(stream)), 20000)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2 * 1024**2)
        self.assertGreater(len(encoded) * 20000, peak * 2)

        config, _ = fixture_manifest()
        config["data_file"] = "selected.json"
        api = SimpleNamespace(
            dataset_info=lambda *a, **kw: SimpleNamespace(
                sha="b" * 40,
                siblings=[SimpleNamespace(size=673_000_000, rfilename="selected.json")]
            ),
            model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=[]),
        )
        expected = materialize(config, api=api, loader=lambda *a, **kw: rows,
                               tokenizer=FixtureTokenizer())
        for payload in [json.dumps(rows, indent=2).encode(),
                        b"\n".join(json.dumps(row).encode() for row in rows)]:
            with patch("huggingface_hub.HfFileSystem") as fs:
                fs.return_value.open.return_value = io.BytesIO(payload)
                actual = materialize(config, api=api, tokenizer=FixtureTokenizer())
            self.assertEqual(actual, expected)  # first-human prompts, indices, pins and hash
        for payload in [b'{"data":[' + encoded + b"]}", b"[" + encoded + b",]",
                        b"[" + encoded, b"[{}] trailing", b"{}{}"]:
            with self.assertRaises((ValueError, WorkloadLimitError)):
                materialize(config, api=api,
                            loader=lambda *a, payload=payload, **kw: iter_json_records(
                                io.BytesIO(payload)
                            ),
                            tokenizer=FixtureTokenizer())
        oversized = RepeatedStream(b'{"padding":"' + b"x" * MAX_ROW_BYTES + b'"}', 1)
        with self.assertRaisesRegex(WorkloadLimitError, "256 KiB"):
            list(iter_json_records(oversized))

    def test_default_sharegpt_filters_invalid_rows_in_a_bounded_frozen_prefix(self):
        import random

        from .workloads import DEFAULT_DATA_FILE, DEFAULT_DATASET, fingerprint

        config, _ = fixture_manifest()
        config.update(repo_id=DEFAULT_DATASET, data_file=DEFAULT_DATA_FILE)
        rows = [
            {"conversations": [{"from": "human", "value": "first prompt"},
                               {"from": "human", "value": "later prompt"}]},
            {"conversations": [{"from": "human", "value": "valid first"},
                               {"from": "gpt", "value": ""}]},
            {"conversations": [{"from": "human", "value": "second prompt"}]},
            {"conversations": [{"from": "tool", "value": "unsupported role"}]},
            {"conversations": [{"from": "gpt", "value": "no human"}]},
            {"conversations": [{"from": "human", "value": "third prompt"}]},
            {"conversations": []},
        ]
        api = SimpleNamespace(
            dataset_info=lambda *a, **kw: SimpleNamespace(
                sha="b" * 40, siblings=[SimpleNamespace(size=673_000_000,
                                                      rfilename=DEFAULT_DATA_FILE)]
            ),
            model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=[]),
        )
        payload = json.dumps(rows).encode()
        # The cap must stop before requesting the following record, including
        # invalid rows in the budget. Both pins and file are frozen in identity.
        def source(*args, **kwargs):
            yield from rows
            raise AssertionError("read beyond the source-row cap")

        with patch("infiro_platform.workloads.MAX_ROWS", len(rows)):
            expected = materialize(config, api=api, loader=source, tokenizer=FixtureTokenizer())
            with patch("huggingface_hub.HfFileSystem") as fs:
                fs.return_value.open.return_value = io.BytesIO(payload)
                actual = materialize(config, api=api, tokenizer=FixtureTokenizer())
        self.assertEqual(actual, expected)
        selected = random.Random(config["seed"]).sample([0, 2, 5], config["num_prompts"])
        self.assertEqual(actual["row_indices"], selected)
        tokenizer = FixtureTokenizer()
        for index, request in zip(selected, actual["requests"], strict=True):
            tokens = tokenizer.encode(rows[index]["conversations"][0]["value"])
            fixed = (tokens * ((config["input_tokens"] + len(tokens) - 1) // len(tokens)))
            self.assertEqual(request["prompt"], tokenizer.decode(fixed[:config["input_tokens"]]))
        self.assertEqual(actual["sampling"], {
            "policy": "default_sharegpt_valid_prefix_v2", "max_source_rows": len(rows),
            "source_rows_scanned": len(rows), "schema_rows_skipped": 4,
            "oversized_rows_skipped": 0,
        })
        self.assertEqual(actual["dataset"]["revision"], "b" * 40)
        self.assertEqual(actual["dataset"]["data_file"], DEFAULT_DATA_FILE)
        self.assertEqual(actual["sha256"], fingerprint(actual))
        changed = dict(actual, sampling=dict(actual["sampling"], schema_rows_skipped=0))
        self.assertNotEqual(actual["sha256"], fingerprint(changed))
        with self.assertRaisesRegex(ValueError, "too few valid prompts"):
            materialize(config, api=api, loader=lambda *a, **kw: rows[1:2],
                        tokenizer=FixtureTokenizer())
        # Custom repositories retain strict schema validation, even if an earlier
        # human turn is usable. Never repair or truncate an invalid conversation.
        config["repo_id"] = "org/custom"
        with self.assertRaisesRegex(ValueError, "Invalid ShareGPT role or empty text"):
            materialize(config, api=api, loader=lambda *a, **kw: rows,
                        tokenizer=FixtureTokenizer())

    def test_default_sharegpt_source_cache_reuses_records_and_preserves_identity(self):
        from .workloads import DEFAULT_DATA_FILE, DEFAULT_DATASET

        config, _ = fixture_manifest()
        config.update(repo_id=DEFAULT_DATASET, data_file=DEFAULT_DATA_FILE)
        rows = [{"conversations": [{"from": "human", "value": f"Prompt {i}"}]}
                for i in range(8)]
        rows[1] = {"conversations": [{"from": "human", "value": ""}]}
        commit = ["b" * 40]
        api = SimpleNamespace(
            dataset_info=lambda *a, **kw: SimpleNamespace(
                sha=commit[0], siblings=[SimpleNamespace(size=1024,
                                                       rfilename=DEFAULT_DATA_FILE)]
            ),
            model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=[]),
        )
        expected = materialize(config, api=api, loader=lambda *a, **kw: rows,
                               tokenizer=FixtureTokenizer())
        cache = Path(self.root.name) / "source-cache"
        stages = []
        with patch.dict(os.environ, {"INFIRO_SHAREGPT_CACHE": str(cache)}):
            with patch("huggingface_hub.HfFileSystem") as fs:
                fs.return_value.open.return_value = io.BytesIO(json.dumps(rows).encode())
                cold = materialize(config, api=api, tokenizer=FixtureTokenizer(),
                                   progress=stages.append)
                fs.return_value.open.assert_called_once()
                fs.assert_called_once_with(token=False)
            with patch("huggingface_hub.HfFileSystem") as fs:
                warm = materialize(config, api=api, tokenizer=FixtureTokenizer(),
                                   progress=stages.append)
                changed = materialize(config | {"seed": 7, "input_tokens": 8},
                                      api=api, tokenizer=FixtureTokenizer())
                fs.assert_not_called()
            self.assertEqual(cold, expected)
            self.assertEqual(warm, expected)
            self.assertEqual(changed["dataset"], expected["dataset"])
            self.assertNotEqual(changed["sha256"], expected["sha256"])
            self.assertEqual(cold["sampling"]["schema_rows_skipped"], 1)
            self.assertTrue(any("Reusing locally cached" in stage for stage in stages))
            commit[0] = "c" * 40
            with patch("huggingface_hub.HfFileSystem") as fs:
                fs.return_value.open.return_value = io.BytesIO(json.dumps(rows).encode())
                updated = materialize(config, api=api, tokenizer=FixtureTokenizer())
                fs.return_value.open.assert_called_once()
            self.assertNotEqual(updated["sha256"], warm["sha256"])
            self.assertEqual(updated["dataset"]["revision"], commit[0])
            for path in cache.glob("*.jsonl"):
                path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "cache failed its integrity check"):
                materialize(config, api=api, tokenizer=FixtureTokenizer())

    def test_default_sharegpt_drains_oversized_rows_and_caches_exclusions(self):
        import random
        import tracemalloc

        from .workloads import DEFAULT_DATA_FILE, DEFAULT_DATASET, iter_json_records

        config, _ = fixture_manifest()
        config.update(repo_id=DEFAULT_DATASET, data_file=DEFAULT_DATA_FILE)
        def human(text):
            return {"conversations": [{"from": "human", "value": text}]}
        rows = [human("first prompt"), human('} [ \" \\ ' * 100000),
                human("second prompt"), human("é" * 60000), human("third prompt")]
        payload = json.dumps(rows, ensure_ascii=False).encode()
        # Discarded records can be arbitrarily larger than retained parser memory.
        stream = io.BytesIO(payload)
        tracemalloc.start()
        try:
            parsed = list(iter_json_records(stream, skip_oversized=True))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2 * 1024**2)
        self.assertIsNone(parsed[1])
        self.assertEqual(parsed[2], rows[2])
        self.assertEqual(parsed[4], rows[4])
        api = SimpleNamespace(
            dataset_info=lambda *a, **kw: SimpleNamespace(
                sha="b" * 40, siblings=[SimpleNamespace(size=len(payload),
                                                      rfilename=DEFAULT_DATA_FILE)]
            ),
            model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=[]),
        )
        cache = Path(self.root.name) / "oversized-cache"
        with patch.dict(os.environ, {"INFIRO_SHAREGPT_CACHE": str(cache)}), patch(
            "infiro_platform.workloads.MAX_ROWS", len(rows)
        ):
            with patch("huggingface_hub.HfFileSystem") as fs:
                fs.return_value.open.return_value = io.BytesIO(payload)
                cold = materialize(config, api=api, tokenizer=FixtureTokenizer())
            with patch("huggingface_hub.HfFileSystem") as fs:
                warm = materialize(config, api=api, tokenizer=FixtureTokenizer())
                fs.assert_not_called()
        self.assertEqual(cold, warm)
        self.assertEqual(cold["row_indices"], random.Random(config["seed"]).sample([0, 2, 4], 3))
        self.assertEqual(cold["sampling"]["source_rows_scanned"], 5)
        self.assertEqual(cold["sampling"]["oversized_rows_skipped"], 2)
        self.assertEqual(cold["sampling"]["schema_rows_skipped"], 0)
        tokenizer = FixtureTokenizer()
        for index, request in zip(cold["row_indices"], cold["requests"], strict=True):
            tokens = tokenizer.encode(rows[index]["conversations"][0]["value"])
            fixed = (tokens * 16)[:config["input_tokens"]]
            self.assertEqual(request["prompt"], tokenizer.decode(fixed))
        # Strict sources still fail. Malformed framing cannot silently resync.
        with self.assertRaisesRegex(ValueError, "256 KiB"):
            list(iter_json_records(io.BytesIO(payload)))
        for broken in [payload[:500000], b'[{"x":"' + b"x" * 300000 + b'"]]']:
            with self.assertRaisesRegex(ValueError, "Unsupported or malformed"):
                list(iter_json_records(io.BytesIO(broken), skip_oversized=True))

    def test_sharegpt_cache_interruption_and_quota(self):
        from .workload_cache import cached_prefix

        cache = Path(self.root.name) / "prefix-cache"
        def interrupted():
            yield {"conversations": []}
            raise OSError("download interrupted")
        with self.assertRaises(OSError):
            cached_prefix(cache, {"revision": "a"}, interrupted, lambda message: None)
        self.assertFalse(list(cache.glob("*.json")))
        data = cached_prefix(cache, {"revision": "a"}, lambda: iter([{"conversations": []}]),
                             lambda message: None)
        self.assertEqual(json.loads(data.read_text()), {"record": {"conversations": []}})
        with patch("infiro_platform.workload_cache.SHAREGPT_DOWNLOAD_BYTES", 4096):
            with self.assertRaisesRegex(ValueError, "cache exceeds its 1 GiB limit"):
                cached_prefix(cache, {"revision": "b"}, lambda: iter([{"conversations": []}]),
                              lambda message: None)

    def test_sharegpt_memory_exhaustion_is_terminal(self):
        from .workloads import MEMORY_ERROR
        config, _ = fixture_manifest()
        def exhausted(*args, **kwargs):
            yield from ()
            raise MemoryError("private source")
        api = SimpleNamespace(
            dataset_info=lambda *a, **kw: SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(size=100, rfilename="data.json")]
            ),
            model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=[]),
        )
        with self.assertRaisesRegex(ValueError, "Shard ShareGPT") as raised:
            materialize(config, api=api, loader=exhausted, tokenizer=FixtureTokenizer())
        self.assertEqual(str(raised.exception), MEMORY_ERROR)
        dataset = Dataset.objects.create(owner=self.alice, name="Source", config=config)
        job = enqueue(self.alice, "dataset", dataset, {}, "memory-terminal")
        fail(job, MEMORY_ERROR)
        job.refresh_from_db()
        self.assertEqual(job.state, "failed")

    def test_large_workload_resource_budget_is_scoped_to_sharegpt(self):
        p = self.prepared()
        random_job = enqueue(self.alice, "prepare", p, {}, "small-random-budget")
        self.assertNotIn("INFIRO_LARGE_WORKLOAD", child_environment(random_job))
        self.assertNotIn("INFIRO_SHAREGPT_CACHE", child_environment(random_job))
        p.choices["workload_mode"] = "sharegpt"
        p.save(update_fields=["choices"])
        sharegpt_job = enqueue(self.alice, "prepare", p, {}, "large-sharegpt-budget")
        environment = child_environment(sharegpt_job)
        self.assertEqual(environment["INFIRO_LARGE_WORKLOAD"], "1")
        from .security import workload_cache_dir

        self.assertEqual(
            environment["INFIRO_SHAREGPT_CACHE"], str(workload_cache_dir(self.alice.id))
        )
        self.assertNotEqual(workload_cache_dir(self.alice.id), workload_cache_dir(self.bob.id))

    def test_worker_counts_persistent_source_cache_in_disk_limit(self):
        from .queue import run_once
        from .security import workload_cache_dir

        prepared = self.prepared()
        prepared.choices["workload_mode"] = "sharegpt"
        prepared.save(update_fields=["choices"])
        job = enqueue(self.alice, "prepare", prepared, {}, "cache-disk-limit")
        (workload_cache_dir(self.alice.id) / "cached-source").write_bytes(b"cached bytes")
        process = Mock(pid=12345)
        process.poll.return_value = None
        with (
            patch("infiro_platform.queue.subprocess.Popen", return_value=process),
            patch("infiro_platform.queue.os.killpg") as kill,
            patch("infiro_platform.queue.SHAREGPT_DISK_BYTES", 1),
        ):
            run_once()
        kill.assert_called_once()
        process.wait.assert_called_once()
        job.refresh_from_db()
        self.assertIn("execution time or 1.25 GiB disk limit", job.error)

    def test_native_random_workload_reproducibility(self):
        config = {
            "model": "org/model",
            "model_revision": "a" * 40,
            "input_tokens": 8,
            "output_tokens": 4,
            "num_prompts": 5,
            "seed": 4,
        }
        a = random_workload(config, tokenizer=FixtureTokenizer())
        b = random_workload(config, tokenizer=FixtureTokenizer())
        self.assertEqual(a, b)
        self.assertEqual(a["dataset"]["repo_id"], "infiro/random")
        self.assertNotEqual(
            a["sha256"],
            random_workload(config | {"seed": 5}, tokenizer=FixtureTokenizer())["sha256"],
        )

    def test_dataset_row_token_and_row_byte_limits_are_explicit(self):
        for setting, value, message in [
            ("MAX_ROWS", 1, "row scan limit"),
            ("MAX_SOURCE_TOKENS", 1, "source token scan limit"),
            ("MAX_ROW_BYTES", 1, "256 KiB"),
        ]:
            with patch("infiro_platform.workloads." + setting, value):
                with self.assertRaisesRegex(ValueError, message):
                    fixture_manifest()

    def test_dataset_access_exception_cannot_expose_provider_secret(self):
        config, _ = fixture_manifest()
        api = SimpleNamespace(dataset_info=Mock(side_effect=RuntimeError("hf-secret-in-error")))
        with self.assertRaisesRegex(ValueError, "loading or access failed") as raised:
            materialize(config, api=api, loader=Mock(), tokenizer=FixtureTokenizer())
        self.assertNotIn("hf-secret-in-error", str(raised.exception))

    def test_worker_timeout_and_disk_limit_kill_only_its_process_group(self):
        from .queue import run_once

        for limit in ("time", "disk"):
            p = self.prepared()
            j = enqueue(self.alice, "prepare", p, {}, "resource-limit-" + limit)
            process = Mock(pid=12345)
            process.poll.return_value = None
            with (
                patch("infiro_platform.queue.subprocess.Popen", return_value=process),
                patch("infiro_platform.queue.os.killpg") as kill,
            ):
                if limit == "time":
                    with patch("infiro_platform.queue.time.monotonic", side_effect=[0, 301]):
                        run_once()
                else:
                    with patch("infiro_platform.queue.DISK_LIMIT", -1):
                        run_once()
                kill.assert_called_once()
                process.wait.assert_called_once()
            j.refresh_from_db()
            self.assertIn("execution time or 256 MiB disk limit", j.error)
            Job.objects.filter(pk=j.pk).update(state="failed")

    def test_api_preparation_is_idempotent_and_invalidates_approved_queue_entry(self):
        p = self.prepared()
        old = approve(self.alice, p.id, p.digest, True)
        c = self.connection(kind="nebius_inference", endpoint="https://example.com/v1")
        data = {
            "recommendation_id": str(p.recommendation_id),
            "choices": Choices(trials=2).model_dump(),
            "connection_refs": {"nebius_inference": str(c.id)},
            "submission_key": "same-request",
        }
        first = self.post("prepared", data)
        second = self.post("prepared", data)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["prepared"]["id"], second.json()["prepared"]["id"])
        p.refresh_from_db()
        self.assertTrue(p.invalidated)
        with self.assertRaises(ValueError):
            perform_study(old)

    def test_api_private_shell_nonce_and_static_alias(self):
        response = self.client.get("/app")
        self.assertIn("nonce-", response["Content-Security-Policy"])
        self.assertIn(b'<script nonce="', response.content)
        self.assertEqual(self.client.get("/app/index.html").status_code, 302)

    def test_changed_connection_cannot_reach_an_isolated_child(self):
        c = self.connection()
        j = enqueue(self.alice, "dataset", c, {"huggingface": str(c.id)}, "version-check")
        Connection.objects.filter(pk=c.pk).update(version=2)
        with self.assertRaisesRegex(ValueError, "Connection changed"):
            child_environment(j)

    def test_iam_transport_renewal_keeps_allocated_request_and_project_immutable(self):
        c = self.connection(kind="nebius_jobs", value="old-iam-token", project_id="project-owned")
        p = self.prepared()
        p.connection_refs = {"nebius_jobs": str(c.id)}
        p.connection_versions = {"nebius_jobs": 1}
        p.digest = digest(binding(p))
        p.save()
        j = approve(self.alice, p.id, p.digest, True)
        Job.objects.filter(pk=j.pk).update(
            controller_run="persisted-reference", state="attention_required"
        )
        response = self.client.put(
            "/app/api/connections/" + str(c.id),
            json.dumps({"credentials": {"iam_token": "new-iam-token"}}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        c.refresh_from_db()
        j.refresh_from_db()
        p.refresh_from_db()
        self.assertEqual(c.version, 2)
        self.assertEqual(j.connection_versions, {"nebius_jobs": 2})
        self.assertTrue(p.invalidated)
        self.assertEqual(self.post("jobs/" + str(j.id) + "/retry", {}).status_code, 202)
        response = self.client.put(
            "/app/api/connections/" + str(c.id),
            json.dumps(
                {
                    "credentials": {"iam_token": "token"},
                    "metadata": {"project_id": "different-project"},
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.delete("/app/api/connections/" + str(c.id)).status_code, 400)

    def test_cancelled_operations_can_retry_but_terminal_studies_need_new_approval(self):
        from .queue import mark_cancelled

        d = Dataset.objects.create(owner=self.alice, name="test", config={"repo_id": "org/data"})
        j = enqueue(self.alice, "dataset", d, {}, "cancel-data")
        mark_cancelled(j)
        d.refresh_from_db()
        self.assertEqual(d.status, "cancelled")
        self.assertEqual(self.post("jobs/" + str(j.id) + "/retry", {}).status_code, 202)
        j.refresh_from_db()
        d.refresh_from_db()
        self.assertEqual(j.state, "queued")
        self.assertEqual(d.status, "validating")
        p = self.prepared()
        study = approve(self.alice, p.id, p.digest, True)
        Job.objects.filter(pk=study.pk).update(state="failed")
        self.assertEqual(self.post("jobs/" + str(study.id) + "/retry", {}).status_code, 400)

    def test_alpaca_manifest_reaches_real_request_preparation(self):
        self.test_selected_sharegpt_file_reaches_real_request_preparation("alpaca")

    def test_messages_manifest_reaches_real_request_preparation(self):
        self.test_selected_sharegpt_file_reaches_real_request_preparation("messages")

    def test_selected_sharegpt_file_reaches_real_request_preparation(self, source_format=None):
        from llm_optimizer.remote.contracts import OptimizationRequest

        from .security import private_dir
        from .services import perform_prepare
        from .workloads import DEFAULT_DATA_FILE, DEFAULT_DATASET, fingerprint

        prepared = seed_prepared(self.alice)
        _, manifest = fixture_manifest()
        manifest["dataset"].update(repo_id=DEFAULT_DATASET, data_file=DEFAULT_DATA_FILE)
        if source_format:
            manifest["dataset"].update(source_format=source_format, adapter_version=1)
        manifest["sha256"] = fingerprint(manifest)
        prepared.dataset = None
        prepared.choices["workload_mode"] = "sharegpt"
        prepared.choices.update(objectives=["mean_ttft_ms"], sampler="RandomSampler", pruner="MedianPruner")
        prepared.payload, prepared.intent, prepared.summary, prepared.digest = {}, {}, {}, ""
        prepared.status = "queued"
        prepared.save()
        job = enqueue(self.alice, "prepare", prepared, {}, "selected-file-real-prepare")
        with (
            patch("infiro_platform.workloads.materialize", return_value=manifest) as sampler,
            patch("hf_model_watch.optimization.preparation.research", return_value={
                "sources": [], "inspection": {"claims": [], "heads": []}
            }) as research,
        ):
            perform_prepare(job)  # Real PreparationInputs, adapter and request writer.
            perform_prepare(job)  # Recovery must reuse the frozen request.
        sampler.assert_called_once()
        research.assert_called_once()
        prepared.refresh_from_db()
        self.assertEqual(prepared.status, "prepared")
        self.assertTrue(prepared.digest)
        directory = private_dir(self.alice.id, job.id)
        request = json.loads((directory / "request.json").read_text())
        intent = json.loads((directory / "intent.json").read_text())
        self.assertEqual(request["workload_manifest"], manifest)
        self.assertEqual(request["optuna"]["objectives"], ["mean_ttft_ms"])
        self.assertEqual(request["optuna"]["sampler"], "RandomSampler")
        self.assertEqual(request["optuna"]["pruner"], "MedianPruner")
        self.assertEqual(intent["context"]["objectives"], ["mean_ttft_ms"])
        self.assertEqual(intent["context"]["workload"]["dataset"]["data_file"], DEFAULT_DATA_FILE)
        self.assertEqual(request["semantic_context"]["workload"]["dataset"]["data_file"],
                         DEFAULT_DATA_FILE)
        self.assertEqual(
            OptimizationRequest.parse(request).to_dict()["workload_manifest"], manifest
        )
        self.assertFalse(Job.objects.filter(kind="study", target_id=prepared.id).exists())

        from llm_optimizer.remote.worker import optimizer_command
        args = optimizer_command({"request": prepared.payload, "framework": "vllm"},
                                 Path(self.root.name) / "controlled-results.json")
        self.assertEqual(args[args.index("--optuna-sampler") + 1], "RandomSampler")
        self.assertEqual(args[args.index("--optuna-pruner") + 1], "MedianPruner")
        self.assertEqual(args[args.index("--optuna-objective") + 1], "mean_ttft_ms")

    def test_optimization_choices_reject_unsupported_combinations(self):
        for metric in ["output_throughput", "mean_e2e_latency_ms", "mean_ttft_ms"]:
            for sampler in ["TPESampler", "RandomSampler", "GridSampler", "GPSampler", "NSGAIISampler"]:
                choices = Choices(objectives=[metric], sampler=sampler, pruner="MedianPruner")
                self.assertEqual(choices.objectives, [metric])
        self.assertEqual(Choices(concurrency=1).objectives, ["p95_e2e_latency_ms", "p95_ttft_ms"])
        self.assertEqual(Choices(concurrency=4).objectives, ["output_throughput", "p95_e2e_latency_ms"])
        self.assertTrue(Choices(concurrency=4, tpe_multivariate=True).tpe_multivariate)
        for invalid in [
            {"pruner": "MedianPruner"}, {"pruner": "ThresholdPruner"},
            {"sampler": "UnknownSampler"}, {"constraints": []},
            {"sampler": "RandomSampler", "tpe_multivariate": True},
            {"objectives": ["mean_e2e_latency_ms", "mean_e2e_latency_ms"]},
            {"objectives": ["p95_ttft_ms", "p95_ttft_ms"]},
            {"objectives": ["unknown_metric"]},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Choices(**invalid)

    def test_approval_rejects_multivariate_setting_dropped_during_preparation(self):
        from .services import finalize_preparation

        with patch("infiro_platform.demo.Choices", side_effect=lambda **kw: Choices(
            tpe_multivariate=True, **kw
        )):
            prepared = seed_prepared(self.alice)
        prepared.payload["optuna"].pop("sampler_kwargs")
        prepared.summary["draft_sha256"] = digest({
            "payload": prepared.payload, "intent": prepared.intent,
            "workload_identity": prepared.summary["workload_identity"]})
        job = enqueue(self.alice, "prepare", prepared, {}, "multivariate-drift")
        with self.assertRaisesRegex(ValueError, "multivariate TPE differs"):
            finalize_preparation(job, prepared, {})

    def test_queue_wait_survives_polling_and_completion(self):
        p = seed_prepared(self.alice)
        job = approve(self.alice, p.id, p.digest, True)
        controller = Mock()
        controller.submit.return_value = {"run_id": "timing-run"}
        controller.status.return_value = {"state": "running", "created_at": "2026-10-26T10:00:00Z",
            "jobs": [{"framework": "vllm", "state": "running"}]}
        controller.results.return_value = {"run_id": "timing-run", "state": "running", "manifests": {}}
        observed = timezone.datetime(2026, 10, 26, 10, 0, 5, tzinfo=timezone.get_default_timezone())
        with patch("infiro_platform.controller.make_controller", return_value=controller), patch("infiro_platform.services.timezone.now", return_value=observed):
            perform_study(job)
            self.assertEqual(job.result["summary"]["framework_timings"]["vllm"]["queue_wait_s"], 5)
            controller.status.return_value["state"] = "completed"
            controller.status.return_value["jobs"][0]["state"] = "completed"
            perform_study(job)
        self.assertEqual(job.result["summary"]["framework_timings"]["vllm"]["queue_wait_s"], 5)
        self.assertIsNone(job.result["summary"]["framework_timings"]["vllm"]["benchmark_s"])

    def test_configuration_validation_is_terminal_and_diagnostics_are_redacted(self):
        from pydantic import ValidationError

        from hf_model_watch.optimization.preparation import PreparationInputs

        from .diagnostics import failure_details, failure_summary

        try:
            PreparationInputs.model_validate({"run_id": "run_test", "dataset": {
                "repo_id": "org/data", "data_file": "https://host/?token=private-secret",
                "private-prompt": "credential-value",
            }})
        except ValidationError as exc:
            details = failure_details(exc)
        else:
            self.fail("Invalid selected-file input was accepted")
        self.assertEqual(details["exceptions"][0]["validation_fields"], [
            {"field": "dataset.data_file", "code": "value_error"},
            {"field": "unrecognized_field", "code": "extra_forbidden"},
        ])
        for secret in ("https://", "private-secret", "private-prompt", "credential-value"):
            self.assertNotIn(secret, json.dumps(details))
        message = failure_summary(details)
        self.assertIn("will not be retried", message)
        prepared = seed_prepared(self.alice)
        job = enqueue(self.alice, "prepare", prepared, {}, "invalid-configuration-terminal")
        Job.objects.filter(pk=job.pk).update(state="running", attempts=1,
                                            progress={"diagnostics": details})
        fail(job, message)
        job.refresh_from_db()
        self.assertEqual(job.state, "failed")
        self.assertEqual(job.attempts, 1)
        self.assertIsNone(claim(kinds=["prepare"]))
        # Even old diagnostics without field details must stop the same retry loop.
        transient = enqueue(self.alice, "prepare", prepared, {}, "transient-still-retries")
        Job.objects.filter(pk=transient.pk).update(state="running", attempts=1)
        fail(transient, "Provider temporarily unavailable")
        transient.refresh_from_db()
        self.assertEqual(transient.state, "queued")
        Job.objects.filter(pk=transient.pk).update(progress={"diagnostics": {"exceptions": [{
            "type": "ValidationError", "module": "pydantic_core._pydantic_core", "frames": []
        }]}})
        fail(transient, "Operation failed (ValidationError).")
        transient.refresh_from_db()
        self.assertEqual(transient.state, "failed")

    def test_recovery_never_regenerates_an_already_prepared_request(self):
        from .services import perform_prepare

        p = self.prepared()
        approve(self.alice, p.id, p.digest, True)
        j = enqueue(self.alice, "prepare", p, {}, "resume-preparation")
        original = p.digest
        with patch("hf_model_watch.optimization.preparation.prepare") as prepare:
            perform_prepare(j)
            prepare.assert_not_called()
        p.refresh_from_db()
        self.assertEqual(p.digest, original)
        self.assertEqual(j.state, "completed")

    def test_provider_failure_preserves_unapproved_draft_and_retry_reuses_it(self):
        from .security import private_dir
        from .services import perform_prepare

        p = seed_prepared(self.alice)
        jobs = self.connection(kind="nebius_jobs", project_id="project-test")
        storage = self.connection(
            kind="storage",
            bucket="private-bucket",
            region="eu-north1",
            secret_version_id="mbsec-storage",
        )
        p.connection_refs = {"nebius_jobs": str(jobs.id), "storage": str(storage.id)}
        p.connection_versions = {"nebius_jobs": jobs.version, "storage": storage.version}
        p.choices["dry_run"] = False
        p.payload, p.intent, p.summary, p.digest, p.status = {}, {}, {}, "", "queued"
        p.save()
        j = enqueue(self.alice, "prepare", p, p.connection_refs, "draft-first")
        directory = private_dir(self.alice.id, j.id)

        def failed_lookup(found):
            self.assertTrue((directory / "request.json").is_file())
            self.assertTrue((directory / "intent.json").is_file())
            p.refresh_from_db()
            self.assertEqual(p.status, "draft")
            self.assertFalse(p.digest)
            self.assertIsNone(p.approved_at)
            raise ValueError("Provider unavailable")

        with (
            patch(
                "hf_model_watch.optimization.preparation.research",
                return_value={"sources": [], "inspection": {"claims": [], "heads": []}},
            ),
            patch(
                "infiro_platform.nebius_provider.pin_secret_references", side_effect=failed_lookup
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Provider unavailable"):
                perform_prepare(j)
        p.refresh_from_db()
        self.assertEqual(p.summary["provider_verification"], "pending")
        with self.assertRaises(ValueError):
            approve(self.alice, p.id, "", True)
        self.assertFalse(Job.objects.filter(kind="study", target_id=p.id).exists())
        original = {
            name: (directory / name).read_bytes()
            for name in ("request.json", "intent.json", "workload.json")
        }
        artifact = Artifact.objects.get(job=j, name="request.json")
        response = self.client.get("/app/api/artifacts/" + str(artifact.id))
        self.assertEqual(response.status_code, 200)
        # Consume the test client's streaming wrapper so request cleanup does not
        # close the PostgreSQL connection inside this TestCase's transaction.
        b"".join(response.streaming_content)
        self.client.force_login(self.bob)
        self.assertEqual(self.client.get("/app/api/artifacts/" + str(artifact.id)).status_code, 404)
        pinned = {
            "nebius_jobs": jobs.metadata,
            "storage": storage.metadata | {"secret_version_id": "mbsecver-pinned"},
        }
        with (
            patch("hf_model_watch.optimization.preparation.prepare") as draft_builder,
            patch("infiro_platform.nebius_provider.pin_secret_references", return_value=pinned),
        ):
            perform_prepare(j)
            draft_builder.assert_not_called()
        p.refresh_from_db()
        self.assertEqual(p.status, "prepared")
        self.assertTrue(p.digest)
        self.assertIsNone(p.approved_at)
        self.assertEqual(
            p.summary["provider_settings"]["storage"]["secret_version_id"], "mbsecver-pinned"
        )
        for name, content in original.items():
            self.assertEqual((directory / name).read_bytes(), content)

    def test_workload_with_known_credential_is_rejected_before_serialization(self):
        p = self.prepared()
        j = enqueue(self.alice, "prepare", p, {}, "sensitive-workload")
        with patch.dict(os.environ, {"HF_TOKEN": "private-token-value"}):
            with self.assertRaisesRegex(ValueError, "contains a configured credential"):
                write_artifact(
                    j, "workload.json", {"requests": [{"prompt": "private-token-value"}]}
                )
        self.assertFalse(Artifact.objects.filter(job=j).exists())


@override_settings(INFIRO_CREDENTIAL_KEYS=TEST_KEY, ALLOWED_HOSTS=["testserver"])
class ControllerIntegrationTests(TransactionTestCase):
    def setUp(self):
        self.root = TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        override = override_settings(INFIRO_PRIVATE_ROOT=Path(self.root.name))
        override.enable()
        self.addCleanup(override.disable)
        self.alice = get_user_model().objects.create_user("alice")
        self.bob = get_user_model().objects.create_user("bob")

    def test_web_preparation_generates_shared_sglang_choices(self):
        import ast

        from llm_optimizer.search_profiles import PREFILL_BUDGETS_1

        with patch("infiro_platform.demo.Choices", side_effect=lambda **kw: Choices(
            frameworks=["sglang"], **kw
        )):
            prepared = seed_prepared(self.alice)
        self.assertEqual(prepared.status, "prepared")
        artifact = Artifact.objects.get(job__target_id=prepared.id, name="request.json")
        from .security import private_dir

        request_path = private_dir(artifact.owner_id, artifact.job_id) / artifact.relative_path
        request = json.loads(request_path.read_text())
        self.assertEqual(request, prepared.payload)
        args = request["server_args"]["sglang"]
        names = [a.partition("=")[0].partition(":")[0] for a in args]
        self.assertEqual(len(names), len(set(names)))
        space = {a.partition("=")[0].partition(":")[0]: ast.literal_eval(a.partition("=")[2])
                 for a in args}
        self.assertEqual(space["cuda_graph_backend_decode"], ["disabled", "full"])
        self.assertEqual(space["cuda_graph_backend_prefill"], ["disabled", "breakable"])
        self.assertEqual(space["attention_backend"], ["triton", "fa3", "flashinfer"])
        self.assertEqual(space["disable_radix_cache*schedule_policy"], [
            [False, "fcfs"], [False, "lpm"], [True, "fcfs"], [True, "lpm"]])
        self.assertNotIn("schedule_policy", names)
        self.assertNotIn("disable_radix_cache", names)
        self.assertEqual(space["max_running_requests"], [1])
        self.assertEqual(space["chunked_prefill_size"], PREFILL_BUDGETS_1)
        self.assertEqual(space["mem_fraction_static"], [0.70, 0.75, 0.80, 0.85])

    def test_controlled_study_retains_settings_through_approval_execution_and_results(self):
        with patch("infiro_platform.demo.Choices", side_effect=lambda **kw: Choices(
            objectives=["mean_ttft_ms"], sampler="TPESampler", tpe_multivariate=True,
            pruner="MedianPruner", **kw
        )):
            prepared = seed_prepared(self.alice)
        self.assertEqual(prepared.payload["optuna"]["sampler_kwargs"], {"multivariate": True})
        from llm_optimizer.remote.worker import optimizer_command
        args = optimizer_command({"request": prepared.payload, "framework": "vllm"},
                                 Path(self.root.name) / "multivariate-results.json")
        self.assertEqual(args[args.index("--optuna-sampler") + 1], "TPESampler")
        self.assertEqual(json.loads(args[args.index("--optuna-sampler-kwargs") + 1]),
                         {"multivariate": True})
        study = approve(self.alice, prepared.id, prepared.digest, True)
        perform_study(study)
        self.assertEqual(study.state, "dry_run")
        self.assertEqual(study.result["optimization_settings"], {
            "objectives": ["mean_ttft_ms"], "sampler": "TPESampler", "pruner": "MedianPruner",
            "sampler_kwargs": {"multivariate": True},
            "trials": 2, "seed": 1})
        study.save()
        perform_study(study)
        self.assertEqual(study.result["optimization_settings"]["pruner"], "MedianPruner")

    def test_real_prepare_controller_dry_run_restart_results_and_two_user_scope(self):
        from .controller import ScopedController

        for user in (self.alice, self.bob):
            p = seed_prepared(user)
            self.assertEqual(p.status, "prepared")
            self.assertEqual(p.summary["optuna"]["trials"], p.choices["trials"])
            self.assertEqual(p.payload["optuna"]["trials"], p.choices["trials"])
            self.assertEqual(
                p.payload["workload_manifest"]["sha256"], p.summary["workload_identity"]["sha256"]
            )
            j = approve(user, p.id, p.digest, True)
            perform_study(j)
            j.save()
            self.assertEqual(j.state, "dry_run")
            self.assertEqual(j.result["summary"]["candidates"], [])
            self.assertEqual(j.result["measurement_status"], "unmeasured_dry_run")
            original = j.controller_run
            perform_study(j)
            self.assertEqual(j.controller_run, original)
            self.assertEqual(ScopedController(j).status(original)["state"], "dry_run")
        a, b = list(Job.objects.filter(kind="study").order_by("created_at"))
        with self.assertRaises(KeyError):
            ScopedController(a).status(b.controller_run)
        self.assertEqual(
            Artifact.objects.filter(owner=self.alice, job=a, name="results.json").count(), 1
        )

    def test_results_reuse_measured_semantics_and_pareto_analysis(self):
        from hf_model_watch.optimization.analysis import analyze
        from hf_model_watch.optimization.contracts import (
            Candidate,
            SelectionPolicy,
            StudyContext,
            StudySummary,
        )

        context = StudyContext(
            model_id="org/model",
            revision="a" * 40,
            gpu_type="gpu-h100-sxm",
            concurrency=4,
            workload={"input_tokens": 16, "output_tokens": 8},
        )
        candidates = [
            Candidate(
                configuration_id=str(i),
                framework="vllm",
                configuration={},
                metrics={"mean_e2e_latency_ms": lat, "output_throughput": tp},
            )
            for i, (lat, tp) in enumerate([(100, 200), (200, 500), (300, 100)])
        ]
        result = analyze(
            StudySummary(
                study_reference="test", context=context, state="completed", candidates=candidates
            ),
            SelectionPolicy(),
        )
        self.assertEqual(len(result.pareto_front), 2)
        self.assertIsNotNone(result.selected)

    @override_settings(INFIRO_LIVE_ENABLED=True)
    def test_mocked_provider_measurements_and_running_cancellation_use_real_controller(self):
        from .controller import ScopedController

        payloads = []
        provider = Mock()
        provider.submit.side_effect = lambda payload: payloads.append(payload) or "aijob-test"
        provider.status.return_value = "completed"
        store = Mock()
        store.progress.return_value = None
        store.put_report.return_value = "s3://private/aggregate.json"

        def manifest(run_id, framework):
            request = payloads[-1]["request"]
            return {
                "run_id": run_id,
                "framework": framework,
                "status": "completed",
                "model": request["model"],
                "revision": request["revision"],
                "resolved_revision": request["revision"],
                "gpu_type": request["gpu_type"],
                "gpu_count": request["gpu_count"],
                "workload": request["client_args"],
                "semantic_context": request["semantic_context"],
                "workload_sha256": request["workload_manifest"]["sha256"],
                "objectives": request["optuna"]["objectives"],
                "constraints": request["constraints"],
                "study_summary": {"trial_state_counts": {"COMPLETE": 2}, "gpu_seconds": 10},
                "test_results": [
                    {
                        "optimization": {"trial_number": i, "state": "COMPLETE"},
                        "config": {"batch_size": i + 1},
                        "results": {
                            "mean_e2e_latency_ms": latency,
                            "mean_ttft_ms": ttft,
                            "p95_e2e_latency_ms": latency * 1.5,
                            "p95_ttft_ms": ttft * 1.5,
                            "output_throughput": throughput,
                            "total_requests": 3,
                            "completed": 3,
                            "failure_rate": 0,
                            "benchmark_valid": True,
                        },
                    }
                    for i, (latency, ttft, throughput) in enumerate(
                        [(100, 30, 200), (150, 10, 300)]
                    )
                ],
            }

        store.manifest.side_effect = manifest
        for cancel in (False, True):
            p = seed_prepared(self.alice)
            p.payload["dry_run"] = False
            p.choices["dry_run"] = False
            p.digest = digest(binding(p))
            p.save()
            j = approve(self.alice, p.id, p.digest, True)
            controller = ScopedController(j, provider=provider, s3_factory=lambda *args: store)
            provider.status.return_value = "running" if cancel else "completed"
            with patch("infiro_platform.controller.make_controller", return_value=controller):
                perform_study(j)
                j.save()
                if cancel:
                    j.cancel_requested = True
                    perform_study(j)
                    self.assertEqual(j.state, "cancelled")
                    provider.cancel.assert_called_with("aijob-test")
                else:
                    self.assertEqual(j.state, "completed")
                    self.assertEqual(j.result["measurement_status"], "measured")
                    self.assertEqual(len(j.result["summary"]["candidates"]), 2)
                    self.assertEqual(len(j.result["analysis"]["pareto_front"]), 2)
