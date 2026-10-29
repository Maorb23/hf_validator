"""Reruns preserve history and always require a new approved review."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from pydantic import ValidationError

from .contracts import Choices
from .demo import seed_prepared
from .models import Job, Prepared
from .security import digest, seal, unseal
from .services import binding, enqueue, perform_prepare


@override_settings(
    INFIRO_CREDENTIAL_KEYS=Fernet.generate_key().decode(), ALLOWED_HOSTS=["testserver"]
)
class RerunTests(TestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        override = override_settings(INFIRO_PRIVATE_ROOT=Path(root.name))
        override.enable()
        self.addCleanup(override.disable)
        self.user = get_user_model().objects.create_user("rerun-user")
        self.client.force_login(self.user)
        self.prepared = seed_prepared(self.user)
        self.job = enqueue(self.user, "study", self.prepared, {}, "original")
        self.job.state = "failed"
        self.job.save()

    def post(self, key="rerun"):
        return self.client.post("/app/api/jobs/" + str(self.job.id) + "/rerun",
                                json.dumps({"submission_key": key}),
                                content_type="application/json")

    def test_clone_preserves_original_and_rechecks_before_approval(self):
        original = self.prepared.digest
        response = self.post()
        self.assertEqual(response.status_code, 202, response.content)
        revised = Prepared.objects.get(pk=response.json()["prepared"]["id"])
        self.assertNotEqual(revised.pk, self.prepared.pk)
        self.assertIsNone(revised.approved_at)
        self.assertEqual(revised.digest, "")
        self.assertEqual(revised.payload, self.prepared.payload)
        self.assertEqual(revised.summary["rerun_of"], str(self.job.id))
        self.assertNotEqual(
            revised.intent["idempotency_key"], self.prepared.intent["idempotency_key"]
        )
        self.assertEqual(self.post().json()["prepared"]["id"], str(revised.id))
        preparation = Job.objects.get(kind="prepare", target_id=revised.id)
        perform_prepare(preparation)
        revised.refresh_from_db()
        self.assertEqual(revised.status, "prepared")
        self.assertIsNone(revised.approved_at)
        self.assertEqual(Job.objects.filter(kind="study").count(), 1)
        self.prepared.refresh_from_db()
        self.assertEqual(self.prepared.digest, original)
        self.assertFalse(self.prepared.invalidated)
        self.job.refresh_from_db()
        self.assertEqual(self.job.state, "failed")
        request = json.loads(json.dumps(revised.payload))
        request["timeout_minutes"] = 120
        edited = self.client.post(
            "/app/api/prepared/" + str(revised.id) + "/request",
            json.dumps({"digest": revised.digest, "request": request}),
            content_type="application/json")
        self.assertEqual(edited.status_code, 201, edited.content)
        self.assertEqual(edited.json()["prepared"]["summary"]["rerun_of"], str(self.job.id))

    def test_owner_and_running_state_are_enforced(self):
        other = get_user_model().objects.create_user("other")
        self.client.force_login(other)
        self.assertEqual(self.post().status_code, 404)
        self.client.force_login(self.user)
        self.job.state = "running"
        self.job.save()
        self.assertEqual(self.post().status_code, 400)

    def test_random_requests_are_rejected_and_history_remains_readable(self):
        with self.assertRaises(ValidationError):
            Choices(workload_mode="random")
        self.prepared.choices["workload_mode"] = "random"
        self.prepared.digest = digest(binding(self.prepared))
        self.prepared.save()
        self.assertEqual(self.post().status_code, 400)
        self.assertEqual(self.client.get("/app/api/jobs/" + str(self.job.id)).status_code, 200)

    def test_removed_session_token_rejected_for_new_connections(self):
        from .models import Connection
        connection = Connection(owner=self.user, kind="storage")
        connection.ciphertext = seal(connection, {"session_token": "legacy"})
        self.assertEqual(unseal(connection)["session_token"], "legacy")
        response = self.client.post("/app/api/connections", json.dumps({
            "name": "storage", "kind": "storage",
            "metadata": {"bucket": "private", "region": "eu-north1"},
            "credentials": {
                "access_key_id": "id", "secret_access_key": "secret", "session_token": "new"
            }
        }), content_type="application/json")
        self.assertEqual(response.status_code, 400)
