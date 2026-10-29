import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from .demo import seed_prepared
from .models import Artifact, Connection, Job, Prepared, Recommendation
from .queue import child_environment, mark_cancelled
from .security import private_dir, seal
from .services import approve, perform_model_selection, perform_prepare, perform_study


@override_settings(
    INFIRO_CREDENTIAL_KEYS=Fernet.generate_key().decode(), ALLOWED_HOSTS=["testserver"]
)
class StudyInputTests(TransactionTestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        override = override_settings(INFIRO_PRIVATE_ROOT=Path(root.name))
        override.enable()
        self.addCleanup(override.disable)
        self.user = get_user_model().objects.create_user("direct-user")
        self.other = get_user_model().objects.create_user("other-user")
        self.client.force_login(self.user)

    def post(self, path, data):
        return self.client.post(
            "/app/api/" + path, json.dumps(data), content_type="application/json"
        )

    def test_direct_model_pins_revision_without_recommendation_or_inference(self):
        body = {
            "model": {"model_id": "org/my-model", "revision": "release"},
            "submission_key": "direct",
        }
        response = self.post("models", body)
        self.assertEqual(response.status_code, 202, response.content)
        rec = Recommendation.objects.get(pk=response.json()["recommendation"]["id"])
        job = Job.objects.get(pk=response.json()["job"]["id"])
        self.assertEqual(job.kind, "model")
        self.assertNotIn("NEBIUS_API_KEY", child_environment(job))
        with (
            patch("huggingface_hub.HfApi") as hub,
            patch("hf_model_watch.recommender.discovery.discover_models") as discover,
        ):
            hub.return_value.model_info.return_value = SimpleNamespace(sha="a" * 40)
            perform_model_selection(job)
            perform_model_selection(job)  # Recovery must not repeat the pin lookup.
            hub.return_value.model_info.assert_called_once_with("org/my-model", revision="release")
            discover.assert_not_called()
        rec.refresh_from_db()
        self.assertEqual(rec.selected, {"model_id": "org/my-model", "revision": "a" * 40})
        self.assertEqual(rec.status, "ready")
        self.assertEqual(self.post("models", body).json()["recommendation"]["id"], str(rec.id))
        self.assertEqual(Recommendation.objects.count(), 1)
        self.assertFalse(Prepared.objects.exists())
        from .demo import fixture_manifest

        connection = Connection(
            owner=self.user,
            name="Inference",
            kind="nebius_inference",
            metadata={"endpoint": "https://api.tokenfactory.nebius.com/v1"},
        )
        connection.ciphertext = seal(connection, {"api_key": "test-inference-key"})
        connection.save()
        response = self.post(
            "prepared",
            {
                "recommendation_id": str(rec.id),
                "choices": {"num_prompts": 3, "input_tokens": 16, "output_tokens": 8},
                "connection_refs": {"nebius_inference": str(connection.id)},
            },
        )
        self.assertEqual(response.status_code, 202, response.content)
        preparation = Job.objects.get(pk=response.json()["job"]["id"])
        with (
            patch(
                "infiro_platform.workloads.materialize",
                side_effect=lambda config, **kwargs: fixture_manifest(model=config["model"], model_revision=config["model_revision"])[1],
            ),
            patch(
                "hf_model_watch.optimization.preparation.research",
                return_value={"sources": [], "inspection": {"claims": [], "heads": []}},
            ),
        ):
            perform_prepare(preparation)
        prepared = Prepared.objects.get(pk=preparation.target_id)
        self.assertEqual(prepared.payload["model"], "org/my-model")
        self.assertEqual(prepared.payload["revision"], "a" * 40)
        study = approve(self.user, prepared.id, prepared.digest, True)
        perform_study(study)
        self.assertEqual(study.state, "dry_run")

    def test_invalid_and_inaccessible_direct_models_and_cancellation(self):
        for model in [{"model_id": "../bad"}, {"model_id": "org/model", "revision": "../bad"}]:
            self.assertEqual(self.post("models", {"model": model}).status_code, 400)
        response = self.post("models", {"model": {"model_id": "org/private"}})
        job = Job.objects.get(pk=response.json()["job"]["id"])
        with patch("huggingface_hub.HfApi") as hub:
            hub.return_value.model_info.side_effect = RuntimeError("private provider detail")
            with self.assertRaisesRegex(ValueError, "Model access or revision lookup failed"):
                perform_model_selection(job)
        mark_cancelled(job)
        self.assertEqual(Recommendation.objects.get(pk=job.target_id).status, "cancelled")

    def edited(self):
        source = seed_prepared(self.user)
        request = copy.deepcopy(source.payload)
        request["timeout_minutes"] = 120
        request["optuna"].update(
            trials=3,
            objectives=["mean_ttft_ms"],
            sampler_kwargs={"multivariate": True, "group": True},
        )
        request["semantic_context"]["objectives"] = ["mean_ttft_ms"]
        request["server_args"]["vllm"] = [
            arg for arg in request["server_args"]["vllm"] if not arg.startswith("max_num_seqs=")
        ]
        request["server_args"]["vllm"].append("max_num_seqs=[8,16]")
        return source, request

    def test_edit_revalidates_new_digest_artifacts_costs_worker_execution(self):
        source, request = self.edited()
        old_digest = source.digest
        response = self.post(
            f"prepared/{source.id}/request", {"digest": old_digest, "request": request}
        )
        self.assertEqual(response.status_code, 201, response.content)
        revised = Prepared.objects.get(pk=response.json()["prepared"]["id"])
        self.assertFalse(revised.digest)
        self.assertIsNone(revised.approved_at)
        source.refresh_from_db()
        self.assertTrue(source.invalidated)
        with self.assertRaisesRegex(ValueError, "stale"):
            approve(self.user, source.id, old_digest, True)
        job = Job.objects.get(kind="prepare", target_id=revised.id)
        with patch("hf_model_watch.optimization.preparation.prepare") as research:
            perform_prepare(job)
            research.assert_not_called()
        revised.refresh_from_db()
        self.assertNotEqual(revised.digest, old_digest)
        self.assertEqual(revised.choices["trials"], 3)
        self.assertEqual(revised.intent["context"]["objectives"], ["mean_ttft_ms"])
        self.assertEqual(revised.payload["semantic_context"], revised.intent["context"])
        self.assertEqual(revised.summary["maximum_gpu_cost_usd"], 2)
        self.assertEqual(revised.payload["server_args"]["vllm"][-1], "max_num_seqs=[8,16]")
        artifact = Artifact.objects.get(job=job, name="request.json")
        saved = json.loads((private_dir(self.user.id, job.id) / artifact.relative_path).read_text())
        self.assertEqual(saved, revised.payload)
        self.assertEqual(source.payload["optuna"]["trials"], 2)
        self.assertFalse(Job.objects.filter(kind="study").exists())
        with self.assertRaises(ValueError):
            approve(self.user, revised.id, old_digest, True)
        study = approve(self.user, revised.id, revised.digest, True)
        perform_study(study)
        self.assertEqual(study.state, "dry_run")
        self.assertTrue(study.result["optimization_settings"]["sampler_kwargs"]["multivariate"])

    def test_editor_rejects_invalid_protected_overbudget_and_unsupported_settings(self):
        source = seed_prepared(self.user)
        mutations = [
            lambda p: p["semantic_context"].update(concurrency=8),
            lambda p: p.update(model="org/other"),
            lambda p: p.update(bucket="other-bucket"),
            lambda p: p.update(images={"vllm": "untrusted:image"}),
            lambda p: p.update(client_args=["num_prompts=1"]),
            lambda p: p.update(gpu_count=8, preset="8gpu-128vcpu-1600gb"),
            lambda p: p["optuna"].update(pruner="MedianPruner"),
            lambda p: p["optuna"].update(sampler="custom.module.Sampler"),
            lambda p: p["optuna"].update(
                sampler="RandomSampler", sampler_kwargs={"multivariate": True}
            ),
            lambda p: p["optuna"].update(sampler_kwargs={"custom": "unsupported"}),
            lambda p: p["server_args"]["vllm"].append("--model=['org/other']"),
            lambda p: p["server_args"]["vllm"].append("speculative_config=['unverified']"),
            lambda p: p["server_args"]["vllm"].append("bad=range(1,999999999)"),
        ]
        for mutation in mutations:
            request = copy.deepcopy(source.payload)
            mutation(request)
            response = self.post(
                f"prepared/{source.id}/request", {"digest": source.digest, "request": request}
            )
            self.assertEqual(response.status_code, 400, response.content)
            source.refresh_from_db()
            self.assertFalse(source.invalidated)
        self.assertEqual(Prepared.objects.count(), 1)

    def test_editor_ownership_stale_and_approved_request_protection(self):
        source, request = self.edited()
        url = f"prepared/{source.id}/request"
        self.client.force_login(self.other)
        self.assertEqual(self.client.get("/app/api/" + url).status_code, 404)
        self.assertEqual(
            self.post(url, {"digest": source.digest, "request": request}).status_code, 404
        )
        self.client.force_login(self.user)
        self.assertEqual(self.post(url, {"digest": "stale", "request": request}).status_code, 400)
        approve(self.user, source.id, source.digest, True)
        self.assertEqual(
            self.post(url, {"digest": source.digest, "request": request}).status_code, 400
        )
