"""SDK messages are constructed for real; all provider I/O is mocked."""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from django.test import SimpleTestCase

from .nebius_provider import NebiusSDKProvider, factory, validate_live_configuration


class SecureProviderTests(SimpleTestCase):
    def test_recovered_status_error_is_cleared_only_for_same_provider_job(self):
        from .nebius_provider import observed_provider_call

        class Provider:
            job = SimpleNamespace(progress={})
            failing = True

            @observed_provider_call
            def status(self, identifier):
                if identifier == "job-a" and self.failing:
                    raise TimeoutError("private upstream details")
                return "running"

        provider = Provider()
        with self.assertRaises(TimeoutError):
            provider.status("job-a")
        self.assertIn("provider_error", provider.job.progress)
        provider.status("job-b")
        self.assertIn("provider_error", provider.job.progress)
        provider.failing = False
        provider.status("job-a")
        self.assertNotIn("provider_error", provider.job.progress)
        self.assertIn("recovered_provider_diagnostics", provider.job.progress)

    def test_payload_permission_failure_is_retained_without_secrets(self):
        import json

        from grpc import StatusCode
        from nebius.aio.service_error import RequestError

        error = RequestError(
            SimpleNamespace(
                code=StatusCode.PERMISSION_DENIED,
                service_errors=[
                    SimpleNamespace(code="PermissionDenied", service="mysterybox-dpl", details=None)
                ],
            )
        )
        with (
            patch.dict(
                os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "private-access-key",
                    "AWS_SECRET_ACCESS_KEY": "private-secret-key",
                },
            ),
            patch.object(self.provider, "_verify_secret", side_effect=error),
        ):
            with self.assertRaises(RequestError):
                self.provider.verify_secrets()
        self.assertIn("Grant payload-read permission", self.job.progress["provider_error"])
        serialized = json.dumps(self.job.progress["provider_diagnostics"])
        self.assertNotIn("private-access-key", serialized)
        self.assertNotIn("private-secret-key", serialized)

    def test_secret_ids_are_pinned_before_approval(self):
        from grpc import StatusCode
        from nebius.aio.service_error import RequestError

        from .nebius_provider import pin_secret_references

        found = {
            "nebius_jobs": SimpleNamespace(metadata={"project_id": "project-owner"}),
            "huggingface": SimpleNamespace(metadata={"secret_version_id": "mbsec-secret123"}),
        }
        version = SimpleNamespace(
            metadata=SimpleNamespace(id="mbsecver-pinned123", parent_id="mbsec-secret123")
        )
        secret = SimpleNamespace(
            metadata=SimpleNamespace(id="mbsec-secret123", parent_id="project-owner"),
            spec=SimpleNamespace(primary_version_id=version.metadata.id),
        )
        not_found = RequestError(SimpleNamespace(code=StatusCode.NOT_FOUND))
        with (
            patch("infiro_platform.nebius_auth.make_sdk") as sdk,
            patch("nebius.api.nebius.mysterybox.v1.SecretVersionServiceClient") as versions,
            patch("nebius.api.nebius.mysterybox.v1.SecretServiceClient") as secrets,
        ):
            versions.return_value.get.return_value.wait.side_effect = [not_found, version]
            secrets.return_value.get.return_value.wait.return_value = secret
            pinned = pin_secret_references(found)
            self.assertEqual(pinned["huggingface"]["secret_version_id"], version.metadata.id)
            self.assertEqual(found["huggingface"].metadata["secret_version_id"], "mbsec-secret123")
            sdk.return_value.sync_close.assert_called_once()
            # Provider permission failures must not be swallowed as wrong-ID errors.
            versions.return_value.get.return_value.wait.side_effect = RequestError(
                SimpleNamespace(code=StatusCode.PERMISSION_DENIED)
            )
            with self.assertRaises(RequestError):
                pin_secret_references(found)
            # IAM INVALID_ARGUMENT is not evidence that a secret ID is a version ID.
            secrets.return_value.get.reset_mock()
            versions.return_value.get.return_value.wait.side_effect = RequestError(
                SimpleNamespace(
                    code=StatusCode.INVALID_ARGUMENT,
                    service_errors=[
                        SimpleNamespace(code="JwtKeyNotExists", service="token-service")
                    ],
                )
            )
            with self.assertRaises(RequestError):
                pin_secret_references(found)
            secrets.return_value.get.assert_not_called()
            # A secret from a different project cannot be pinned.
            versions.return_value.get.return_value.wait.side_effect = [not_found, version]
            secret.metadata.parent_id = "project-other"
            with self.assertRaises(ValueError):
                pin_secret_references(found)

    def setUp(self):
        self.job = SimpleNamespace(
            id=uuid4(), owner_id=7, owner=SimpleNamespace(id=7), controller_run="run123"
        )
        self.found = {
            "nebius_jobs": SimpleNamespace(metadata={"project_id": "project-user"}),
            "storage": SimpleNamespace(
                metadata={
                    "secret_version_id": "mbse-storage",
                    "bucket": "private-bucket",
                    "region": "eu-north1",
                }
            ),
            "huggingface": SimpleNamespace(metadata={"secret_version_id": "mbse-hf"}),
        }
        self.payload = {
            "model": "org/model",
            "revision": "a" * 40,
            "frameworks": ["vllm"],
            "gpu_type": "gpu-h100-sxm",
            "gpu_count": 1,
            "preset": "1gpu-16vcpu-200gb",
            "images": {"vllm": "registry/image@sha256:" + "a" * 64},
            "bucket": "private-bucket",
            "prefix": "infiro/users/7/prepared/test",
            "dry_run": False,
            "timeout_minutes": 60,
        }
        self.prepared = SimpleNamespace(
            id="test", choices={"frameworks": ["vllm"]}, payload=self.payload,
            connection_refs={}, summary={}
        )
        self.provider = object.__new__(NebiusSDKProvider)
        self.provider.job = self.job
        self.provider.prepared = self.prepared
        self.provider.found = self.found
        self.provider.project_id = "project-user"
        self.provider.sdk = Mock()
        self.provider.client = Mock()
        self.provider._verified = False

    def test_sdk_spec_contains_pinned_secret_references_and_no_secret_values(self):
        secrets = {
            "AWS_ACCESS_KEY_ID": "private-access-id",
            "AWS_SECRET_ACCESS_KEY": "private-storage-secret",
            "HF_TOKEN": "private-hf-token",
            "NEBIUS_IAM_TOKEN": "private-iam-token",
            "NEBIUS_API_KEY": "private-inference-key",
        }
        with patch.dict(os.environ, secrets):
            spec = self.provider.build_spec(
                {"run_id": "run123", "framework": "vllm", "request": self.payload}
            )
        raw = spec.SerializeToString()
        for value in secrets.values():
            self.assertNotIn(value.encode(), raw)
        fields = {entry.name: entry for entry in spec.environment_variables}
        self.assertEqual(fields["HF_TOKEN"].mysterybox_secret.version_id, "mbse-hf")
        self.assertEqual(
            fields["AWS_SECRET_ACCESS_KEY"].mysterybox_secret.version_id, "mbse-storage"
        )
        self.assertFalse(fields["HF_TOKEN"].value)
        self.assertNotIn("NEBIUS_IAM_TOKEN", fields)
        self.assertNotIn("NEBIUS_API_KEY", fields)
        import json
        import shlex

        self.assertEqual(shlex.split(spec.args)[0], "-c")
        self.assertIn("run_worker(payload)", shlex.split(spec.args)[1])
        self.assertNotIn("LLM_OPTIMIZER_PAYLOAD_ZLIB_B64", fields)
        ref = json.loads(fields["INFIRO_WORKER_PAYLOAD_REF"].value)
        self.assertEqual(ref["bucket"], "private-bucket")
        self.assertTrue(ref["key"].startswith("infiro/users/7/prepared/test/run123/vllm/launch-"))
        self.assertEqual(spec.restart_attempts, 0)

    def test_live_images_and_worker_secret_versions_are_mandatory(self):
        with self.assertRaises(ValueError):
            validate_live_configuration(self.prepared, self.found, {"vllm": "mutable:latest"})
        self.found["storage"].metadata.pop("secret_version_id")
        with self.assertRaises(ValueError):
            validate_live_configuration(self.prepared, self.found, self.payload["images"])

    def test_shared_wandb_uses_existing_tracker_environment_and_secret_reference(self):
        self.prepared.summary["tracking"] = {
            "project": "shared-studies", "entity": "team", "secret_version_id": "mbsecver-wandb"
        }
        spec = self.provider.build_spec(
            {"run_id": "run123", "framework": "vllm", "request": self.payload}
        )
        fields = {entry.name: entry for entry in spec.environment_variables}
        self.assertEqual(fields["WANDB_PROJECT"].value, "shared-studies")
        self.assertEqual(fields["WANDB_ENTITY"].value, "team")
        self.assertEqual(fields["WANDB_API_KEY"].mysterybox_secret.version_id, "mbsecver-wandb")
        self.assertFalse(fields["WANDB_API_KEY"].value)
        self.assertIn("user-7", fields["WANDB_TAGS"].value)
        self.assertIn(self.job.id.hex, fields["WANDB_TAGS"].value)
        with patch.object(self.provider, "_verify_secret") as verify:
            self.provider.verify_secrets()
        self.assertEqual(verify.call_args.kwargs, {"required_keys": ("WANDB_API_KEY",)})

    def test_secret_project_and_payload_match_are_verified(self):
        version = SimpleNamespace(metadata=SimpleNamespace(parent_id="mbsec-owned"))
        secret = SimpleNamespace(metadata=SimpleNamespace(parent_id="project-user"))
        payload = SimpleNamespace(
            version_id="mbse-hf",
            data=[SimpleNamespace(key="HF_TOKEN", string_value="expected-token")],
        )
        with (
            patch("nebius.api.nebius.mysterybox.v1.SecretVersionServiceClient") as versions,
            patch("nebius.api.nebius.mysterybox.v1.SecretServiceClient") as secrets,
            patch("nebius.api.nebius.mysterybox.v1.PayloadServiceClient") as values,
        ):
            versions.return_value.get.return_value.wait.return_value = version
            secrets.return_value.get.return_value.wait.return_value = secret
            values.return_value.get.return_value.wait.return_value = payload
            self.provider._verify_secret(self.found["huggingface"], {"HF_TOKEN": "expected-token"})
            with self.assertRaises(ValueError):
                self.provider._verify_secret(
                    self.found["huggingface"], {"HF_TOKEN": "different-token"}
                )
            secret.metadata.parent_id = "another-project"
            with self.assertRaises(ValueError):
                self.provider._verify_secret(
                    self.found["huggingface"], {"HF_TOKEN": "expected-token"}
                )

    def test_provider_status_and_cancellation_enforce_remote_ownership(self):
        remote = SimpleNamespace(
            metadata=SimpleNamespace(
                id="aijob-test",
                name=self.provider.name("run123", "vllm"),
                parent_id="project-user",
                labels={"infiro-job": self.job.id.hex, "infiro-user": "7"},
            ),
            status=SimpleNamespace(state=3),
        )
        self.provider.client.get.return_value.wait.return_value = remote
        self.assertEqual(self.provider.status("aijob-test"), "running")
        self.provider.cancel("aijob-test")
        self.provider.client.cancel.assert_called_once()
        remote.metadata.labels["infiro-user"] = "8"
        with self.assertRaises(ValueError):
            self.provider.cancel("aijob-test")
        self.assertEqual(self.provider.client.cancel.call_count, 1)

    def test_portable_ownership_labels_keep_user_and_study_isolation(self):
        labels = self.provider.ownership_labels()
        for key, value in labels.items():
            self.assertRegex(key, r"^[a-z][a-z0-9]{0,62}$")
            self.assertRegex(value, r"^[a-z][a-z0-9]{0,62}$")
        remote = SimpleNamespace(
            metadata=SimpleNamespace(
                parent_id="project-user",
                labels=labels.copy(),
                name=self.provider.name("run123", "vllm"),
                id="aijob-test",
            )
        )
        self.assertIs(self.provider._check_owned(remote), remote)
        remote.metadata.labels["infirouser"] = "user8"
        with self.assertRaises(ValueError):
            self.provider._check_owned(remote)

        # Legacy labels cannot override an explicitly conflicting current owner.
        remote.metadata.labels.update({"infiro-job": self.job.id.hex, "infiro-user": "7"})
        with self.assertRaises(ValueError):
            self.provider._check_owned(remote)
        remote.metadata.labels = labels.copy()
        remote.metadata.labels["infirojob"] = "job" + uuid4().hex
        with self.assertRaises(ValueError):
            self.provider._check_owned(remote)

    def test_unlabelled_jobs_require_exact_study_project_framework_and_id(self):
        remote = SimpleNamespace(
            metadata=SimpleNamespace(
                parent_id="project-user",
                name=self.provider.name("run123", "vllm"),
                id="aijob-test",
                labels={},
            )
        )
        self.provider.client.get.return_value.wait.return_value = remote
        self.assertIs(self.provider.get("aijob-test"), remote)
        remote.metadata.id = "aijob-another"
        with self.assertRaises(ValueError):
            self.provider.get("aijob-test")
        remote.metadata.id = "aijob-test"
        for field, value in (
            ("parent_id", "project-other"),
            ("name", "infiro-" + uuid4().hex + "-vllm"),
            ("name", self.provider.name("run123", "sglang")),
        ):
            original = getattr(remote.metadata, field)
            setattr(remote.metadata, field, value)
            with self.assertRaises(ValueError):
                self.provider._check_owned(remote)
            setattr(remote.metadata, field, original)
        remote.metadata.labels = {"infiro-user": "8"}
        with self.assertRaises(ValueError):
            self.provider._check_owned(remote)
        remote.metadata.labels = {}
        self.provider.client.get_by_name.return_value.wait.return_value = remote
        self.assertEqual(self.provider.find_by_name("run123", "vllm"), "aijob-test")
        with self.assertRaises(ValueError):
            self.provider.find_by_name("another-run", "vllm")
        with self.assertRaises(ValueError):
            self.provider.find_by_name("run123", "sglang")

    def test_submission_reconciles_by_name_before_creating(self):
        with (
            patch.object(self.provider, "verify_secrets"),
            patch.object(self.provider, "find_by_name", return_value="aijob-alreadyaccepted"),
        ):
            self.assertEqual(
                self.provider.submit(
                    {"run_id": "run123", "framework": "vllm", "request": self.payload}
                ),
                "aijob-alreadyaccepted",
            )
        self.provider.client.create.assert_not_called()

    def test_storage_factory_enforces_approved_bucket_prefix_and_run(self):
        with (
            patch("infiro_platform.services.connections", return_value=self.found),
            patch("boto3.client", return_value=Mock()),
            patch("infiro_platform.nebius_provider.NebiusSDKProvider"),
            patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret"}),
        ):
            _, store_type = factory(self.job, self.prepared)
            with self.assertRaises(ValueError):
                store_type("another-bucket", self.payload["prefix"])
            with self.assertRaises(ValueError):
                store_type("private-bucket", "infiro/users/8")
            store = store_type("private-bucket", self.payload["prefix"])
            self.assertTrue(
                store.key("run123", "vllm", "manifest.json").startswith(
                    self.payload["prefix"] + "/run123/"
                )
            )
            for run, framework, name in [
                ("another-run", "vllm", "manifest.json"),
                ("run123", "unknown", "manifest.json"),
                ("run123", "vllm", "../other.json"),
            ]:
                with self.assertRaises(ValueError):
                    store.key(run, framework, name)
