import copy
import hashlib
import io
import json
import os
import random
import shlex
import socket
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from django.test import SimpleTestCase, override_settings

from .nebius_provider import NebiusSDKProvider
from .payload_transport import REFERENCE_ENV, bootstrap_source, load_and_run, payload_blob


class PayloadTransportTests(SimpleTestCase):
    def setUp(self):
        self.job = SimpleNamespace(id=uuid4(), owner_id=7, controller_run="run123", progress={})
        self.request = {
            "model": "org/model",
            "revision": "a" * 40,
            "frameworks": ["vllm"],
            "images": {"vllm": "registry/image@sha256:" + "a" * 64},
            "gpu_type": "gpu-h100-sxm",
            "gpu_count": 1,
            "preset": "1gpu-16vcpu-200gb",
            "bucket": "private-bucket",
            "prefix": "infiro/users/7/prepared/test",
            "dry_run": False,
        }
        self.prepared = SimpleNamespace(
            id="test", payload=self.request, choices={"frameworks": ["vllm"]}, summary={}
        )
        self.payload = {"run_id": "run123", "framework": "vllm", "request": self.request}
        self.provider = object.__new__(NebiusSDKProvider)
        self.provider.job = self.job
        self.provider.prepared = self.prepared
        self.provider.project_id = "project-test"
        self.provider.found = {
            "nebius_jobs": SimpleNamespace(metadata={"project_id": "project-test"}),
            "storage": SimpleNamespace(
                metadata={"region": "eu-north1", "secret_version_id": "mbsecver-test"}
            ),
        }
        self.provider.client = Mock()
        self.provider.sdk = Mock()
        self.alias_verifier = patch(
            "infiro_platform.image_aliases.verified_alias", return_value="registry/image:verified"
        )
        self.alias_verifier.start()
        self.addCleanup(self.alias_verifier.stop)

    def fetch(self, raw, ref, *, loader=load_and_run):
        client = Mock()
        body = io.BytesIO(raw)
        client.get_object.return_value = {"Body": body, "ContentLength": len(raw)}
        with (
            patch.dict(
                os.environ,
                {
                    REFERENCE_ENV: json.dumps(ref),
                    "AWS_ACCESS_KEY_ID": "fake-id",
                    "AWS_SECRET_ACCESS_KEY": "fake-secret",
                    "AWS_DEFAULT_REGION": "eu-north1",
                },
            ),
            patch("boto3.client", return_value=client),
            patch(
                "llm_optimizer.remote.worker.run_worker", return_value={"status": "completed"}
            ) as worker,
        ):
            try:
                result = loader()
            except Exception:
                worker.assert_not_called()
                raise
        client.get_object.assert_called_once_with(Bucket=ref["bucket"], Key=ref["key"])
        self.assertTrue(body.closed)
        return result, worker

    def test_large_workload_is_referenced_and_recovered_by_standalone_bootstrap(self):
        from llm_optimizer.hf_workload import fingerprint

        from .demo import fixture_manifest

        _, manifest = fixture_manifest()
        rng = random.Random(9)
        manifest["requests"] = [
            {"prompt": rng.randbytes(2000).hex(), "prompt_len": 512, "output_len": 128}
            for _ in range(100)
        ]
        manifest["row_indices"] = list(range(100))
        manifest["sha256"] = fingerprint(manifest)
        self.request["workload_manifest"] = manifest
        raw, ref = payload_blob(self.payload, self.prepared, self.job)
        self.assertGreater(len(zlib.compress(raw)), 48_000)
        spec = self.provider.build_spec(self.payload)
        fields = {v.name: v for v in spec.environment_variables}
        self.assertEqual(json.loads(fields[REFERENCE_ENV].value), ref)
        self.assertLess(len(fields[REFERENCE_ENV].value), 4096)
        self.assertNotIn("LLM_OPTIMIZER_PAYLOAD_ZLIB_B64", fields)
        self.assertNotIn(manifest["requests"][0]["prompt"], spec.args)
        self.assertNotIn(manifest["requests"][0]["prompt"].encode(), spec.SerializeToString())
        arguments = shlex.split(spec.args)
        self.assertEqual(arguments[0], "-c")
        namespace = {"__name__": "test_bootstrap"}
        exec(compile(arguments[1], "bootstrap", "exec"), namespace)
        original = socket.socket.connect
        result, worker = self.fetch(raw, ref, loader=namespace["load_and_run"])
        self.assertEqual(result, 0)
        self.assertIs(socket.socket.connect, original)
        self.assertEqual(worker.call_args.args[0]["request"]["workload_manifest"], manifest)

    def test_upload_precedes_creation_and_duplicate_submission_reuses_job(self):
        raw, ref = payload_blob(self.payload, self.prepared, self.job)
        events = []
        store = Mock()
        store.key.return_value = ref["key"]

        def upload(path, run, framework, name):
            self.assertEqual(Path(path).read_bytes(), raw)
            self.assertEqual(
                (run, framework, name), ("run123", "vllm", ref["key"].rsplit("/", 1)[1])
            )
            events.append("upload")

        store.upload.side_effect = upload
        self.provider.payload_storage_factory = Mock(return_value=store)
        operation = Mock(resource_id="aijob-test")
        self.provider.client.create.side_effect = lambda *a, **k: (
            events.append("create") or Mock(wait=Mock(return_value=operation))
        )
        with (
            TemporaryDirectory() as root,
            override_settings(INFIRO_PRIVATE_ROOT=Path(root)),
            patch.object(self.provider, "verify_secrets"),
            patch.object(self.provider, "find_by_name", return_value=None) as finder,
        ):
            self.assertEqual(self.provider.submit(self.payload), "aijob-test")
            finder.return_value = "aijob-test"
            self.assertEqual(self.provider.submit(self.payload), "aijob-test")
        self.assertEqual(events, ["upload", "create"])
        self.assertEqual(
            self.provider.client.create.call_args.args[0].spec.image, "registry/image:verified"
        )
        self.assertIn("@sha256:", self.payload["request"]["images"]["vllm"])
        self.assertEqual(
            dict(self.provider.client.create.call_args.args[0].metadata.labels),
            {},
        )
        self.provider.payload_storage_factory.assert_called_once_with(ref["bucket"], ref["prefix"])

    def test_upload_failure_cannot_create_job(self):
        _, ref = payload_blob(self.payload, self.prepared, self.job)
        store = Mock()
        store.key.return_value = ref["key"]
        store.upload.side_effect = OSError("private provider detail")
        self.provider.payload_storage_factory = Mock(return_value=store)
        with (
            TemporaryDirectory() as root,
            override_settings(INFIRO_PRIVATE_ROOT=Path(root)),
            patch.object(self.provider, "verify_secrets"),
            patch.object(self.provider, "find_by_name", return_value=None),
        ):
            with self.assertRaises(OSError):
                self.provider.submit(self.payload)
        self.provider.client.create.assert_not_called()
        self.assertNotIn("private provider detail", json.dumps(self.job.progress))

    def test_image_verification_failure_blocks_upload_and_creation(self):
        with (
            patch.object(self.provider, "verify_secrets"),
            patch.object(self.provider, "find_by_name", return_value=None),
            patch.object(self.provider, "stage_payload") as stage,
            patch(
                "infiro_platform.image_aliases.verified_alias",
                side_effect=ValueError("changed tag"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "changed tag"):
                self.provider.submit(self.payload)
        stage.assert_not_called()
        self.provider.client.create.assert_not_called()

    def test_provider_rejection_records_actual_request_metadata_and_message(self):
        from grpc import StatusCode
        from nebius.aio.service_error import RequestError
        from nebius.api.nebius.ai.v1 import CreateJobRequest

        _, ref = payload_blob(self.payload, self.prepared, self.job)
        store = Mock()
        store.key.return_value = ref["key"]
        self.provider.payload_storage_factory = Mock(return_value=store)
        error = RequestError(
            SimpleNamespace(
                code=StatusCode.INVALID_ARGUMENT,
                request_id="request-123",
                trace_id="trace-123",
                service_errors=[
                    SimpleNamespace(
                        code="BadRequest",
                        service="compute",
                        details=SimpleNamespace(
                            field="bad_request",
                            value=SimpleNamespace(
                                violations=[
                                    SimpleNamespace(
                                        field="metadata.labels",
                                        message="Label key failed provider validation",
                                    )
                                ]
                            ),
                        ),
                    )
                ],
            )
        )
        self.provider.client.create.side_effect = error
        with (
            TemporaryDirectory() as root,
            override_settings(INFIRO_PRIVATE_ROOT=Path(root)),
            patch.object(self.provider, "verify_secrets"),
            patch.object(self.provider, "find_by_name", return_value=None),
        ):
            with self.assertRaises(RequestError):
                self.provider.submit(self.payload)
        wire = CreateJobRequest.FromString(
            self.provider.client.create.call_args.args[0].SerializeToString()
        )
        self.assertEqual(dict(wire.metadata.labels), {})
        details = self.job.progress["provider_diagnostics"]["submit"]
        self.assertEqual(details["submitted_metadata"]["label_count"], 0)
        self.assertEqual(
            details["label_validation_messages"], ["Label key failed provider validation"]
        )
        self.assertEqual(details["exceptions"][0]["request_id"], "request-123")
        self.provider.client.create.assert_called_once()

    def test_label_validation_message_is_narrowly_scoped_and_redacted(self):
        from grpc import StatusCode
        from nebius.aio.service_error import RequestError

        from .diagnostics import submission_validation_details

        message = "Labels must match ^[a-z]+$: hf-sensitive-secret Bearer generated-access-token https://example.com/?signed=secret"
        error = RequestError(
            SimpleNamespace(
                code=StatusCode.INVALID_ARGUMENT,
                service_errors=[
                    SimpleNamespace(
                        details=SimpleNamespace(
                            field="bad_request",
                            value=SimpleNamespace(
                                violations=[
                                    SimpleNamespace(field="metadata.labels", message=message),
                                    SimpleNamespace(
                                        field="spec.args", message="private prompt text"
                                    ),
                                ]
                            ),
                        )
                    )
                ],
            )
        )
        with patch.dict(os.environ, {"HF_TOKEN": "hf-sensitive-secret"}):
            details = submission_validation_details(error, {"label_count": 0})
        text = json.dumps(details)
        for private in (
            "hf-sensitive-secret",
            "generated-access-token",
            "https://",
            "signed=secret",
            "private prompt text",
        ):
            self.assertNotIn(private, text)
        self.assertIn("Labels must match", text)
        self.assertNotIn(
            "label_validation_messages", submission_validation_details(error, {"label_count": 1})
        )

    def test_internal_image_label_failure_has_actionable_summary(self):
        from .diagnostics import failure_summary

        details = {
            "exceptions": [
                {"type": "RequestError", "provider_errors": [{"fields": ["metadata.labels"]}]}
            ],
            "submitted_metadata": {"label_count": 0, "image_reference_length": 135},
            "label_validation_messages": ["label value length (135) exceeds maximum value (64)"],
        }
        self.assertIn("Nebius Jobs-to-Compute", failure_summary(details))
        for metadata in (
            {"label_count": 1, "image_reference_length": 135},
            {"label_count": 0, "image_reference_length": 137},
            {"label_count": 0},
        ):
            self.assertNotIn(
                "Nebius Jobs-to-Compute",
                failure_summary(details | {"submitted_metadata": metadata}),
            )

    def test_reference_ownership_and_approved_payload_cannot_change(self):
        altered = copy.deepcopy(self.payload)
        altered["request"]["model"] = "org/other-model"
        with self.assertRaisesRegex(ValueError, "approved configuration"):
            payload_blob(altered, self.prepared, self.job)
        other = SimpleNamespace(owner_id=8, controller_run="run123")
        with self.assertRaisesRegex(ValueError, "authorized prefix"):
            payload_blob(self.payload, self.prepared, other)
        with self.assertRaisesRegex(ValueError, "authorized scope"):
            payload_blob(self.payload | {"run_id": "another-run"}, self.prepared, self.job)
        with self.assertRaisesRegex(ValueError, "16 MiB"):
            with patch("infiro_platform.payload_transport.MAX_PAYLOAD_BYTES", 32):
                payload_blob(self.payload, self.prepared, self.job)

    def test_tampering_limits_and_cross_study_payloads_never_start_worker(self):
        raw, ref = payload_blob(self.payload, self.prepared, self.job)
        cases = [
            (raw[:-1] + b"!", ref),
            (raw, ref | {"size_bytes": 17 * 1024 * 1024}),
            (raw, ref | {"key": "infiro/users/8/private.json"}),
            (raw, ref | {"size_bytes": len(raw) + 1}),
        ]
        wrong = json.loads(raw)
        wrong["run_id"] = "different"
        wrong_raw = json.dumps(wrong).encode()
        wrong_sha = hashlib.sha256(wrong_raw).hexdigest()
        cases.append(
            (
                wrong_raw,
                ref
                | {
                    "sha256": wrong_sha,
                    "size_bytes": len(wrong_raw),
                    "key": ref["key"].replace(ref["sha256"], wrong_sha),
                },
            )
        )
        original = socket.socket.connect
        for data, reference in cases:
            with (
                self.subTest(reference=reference),
                patch("llm_optimizer.remote.worker.run_worker") as worker,
            ):
                with self.assertRaises(ValueError):
                    self.fetch(data, reference)
                worker.assert_not_called()
            self.assertIs(socket.socket.connect, original)

    def test_bootstrap_is_static_and_contains_no_platform_runtime_import(self):
        source = bootstrap_source()
        self.assertNotIn("from infiro_platform", source)
        self.assertNotIn("from django", source)
        self.assertIn("Worker payload bootstrap failed", source)

    def test_bootstrap_failure_reports_safe_frames_without_exception_message(self):
        from contextlib import redirect_stderr

        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {REFERENCE_ENV: "private-invalid-json"}),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as stopped,
        ):
            exec(compile(bootstrap_source(), "bootstrap", "exec"), {"__name__": "__main__"})
        self.assertEqual(stopped.exception.code, 2)
        output = stderr.getvalue()
        self.assertNotIn("private-invalid-json", output)
        details = json.loads(output.splitlines()[-1])["bootstrap_diagnostics"]
        self.assertEqual(details["type"], "JSONDecodeError")
        self.assertTrue(details["frames"])
