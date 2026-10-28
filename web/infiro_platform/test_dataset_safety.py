import builtins
import copy
import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from .contracts import DatasetConfig
from .dataset_source import dataset_id
from .models import Dataset, Job
from .workload_runtime import verify_frozen_workload


class DatasetSourceTests(SimpleTestCase):
    def test_only_repository_urls_are_accepted_without_worker_imports(self):
        original = builtins.__import__

        def no_optimizer(name, *args, **kwargs):
            if name == "llm_optimizer" or name.startswith("llm_optimizer."):
                raise ModuleNotFoundError(name=name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_optimizer):
            config = DatasetConfig(
                repo_id="https://huggingface.co/datasets/org/data",
                model="org/model",
                data_file="subdir/test.json",
            )
        self.assertEqual(config.repo_id, "org/data")
        for value in (
            "https://evil.test/datasets/org/data",
            "http://127.0.0.1/data",
            "https://huggingface.co/datasets/org/data?token=secret",
            "https://user:secret@huggingface.co/datasets/org/data",
            "https://huggingface.co/datasets/org/data/blob/main/test.json",
            "../data",
            "org/../data",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dataset_id(value)

    def test_explicit_file_selection_rejects_unimplemented_filters(self):
        base = dict(repo_id="org/data", model="org/model", data_file="test.json")
        for changes in (
            {"data_file": None},
            {"data_file": "../test.json"},
            {"split": "test"},
            {"split": "train[:10]"},
            {"config": "chat"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DatasetConfig(**(base | changes))
        self.assertEqual(DatasetConfig(**base).data_file, "test.json")


class DatasetAPITests(TestCase):
    def test_missing_optimizer_is_a_safe_503_and_queues_nothing(self):
        self.client.force_login(User.objects.create_user(username="dataset-dependency"))
        with patch(
            "infiro_platform.dataset_source.require_optimizer",
            side_effect=ModuleNotFoundError("private filesystem path", name="llm_optimizer"),
        ):
            response = self.client.post(
                "/app/api/datasets",
                json.dumps(
                    {
                        "config": {
                            "repo_id": "org/data",
                            "model": "org/model",
                            "data_file": "test.json",
                        }
                    }
                ),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("both the web and worker", response.json()["error"])
        self.assertNotIn("private filesystem", response.content.decode())
        self.assertFalse(Dataset.objects.exists())
        self.assertFalse(Job.objects.exists())


class FrozenRuntimeTests(SimpleTestCase):
    def payload(self, framework="vllm"):
        from .demo import fixture_manifest

        _, manifest = fixture_manifest()
        return {
            "run_id": "check",
            "framework": framework,
            "request": {
                "model": manifest["tokenizer"],
                "revision": manifest["tokenizer_revision"],
                "frameworks": [framework],
                "images": {framework: "registry/image@sha256:" + "a" * 64},
                "gpu_type": "gpu-h100-sxm",
                "gpu_count": 1,
                "bucket": "private-bucket",
                "prefix": "test",
                "workload_manifest": manifest,
            },
        }

    def test_real_optimizer_reads_identical_frozen_requests_for_every_framework(self):
        for framework in ("vllm", "sglang", "trtllm"):
            with self.subTest(framework=framework):
                payload = self.payload(framework)
                expected = copy.deepcopy(payload)
                result = verify_frozen_workload(payload)
                self.assertEqual(result["status"], "verified")
                self.assertEqual(
                    result["sha256"], payload["request"]["workload_manifest"]["sha256"]
                )
                self.assertEqual(result["request_count"], 3)
                self.assertEqual(payload, expected)

    def test_old_image_and_random_fallback_fail_closed(self):
        payload = self.payload()
        with patch.dict("llm_optimizer.predefined.client.CLIENT_CONFIGS", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Worker image cannot consume"):
                verify_frozen_workload(payload)
        with patch("llm_optimizer.bench_client.get_dataset", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Worker image cannot consume"):
                verify_frozen_workload(payload)
        with patch(
            "llm_optimizer.remote.worker.optimizer_command",
            return_value=["benchmark", "--client-args", "dataset_name=random"],
        ):
            with self.assertRaisesRegex(ValueError, "Worker image cannot consume"):
                verify_frozen_workload(payload)

    def test_tampered_manifest_is_rejected_and_no_manifest_skips_probe(self):
        payload = self.payload()
        payload["request"]["workload_manifest"]["requests"][0]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "Worker image cannot consume"):
            verify_frozen_workload(payload)
        self.assertIsNone(verify_frozen_workload({"request": {}}))
