import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from pydantic import ValidationError

from .contracts import Choices
from .demo import seed_prepared
from .models import Job, Prepared, Recommendation
from .queue import fail
from .request_editing import revise_request
from .security import digest
from .services import binding


class DSparkChoicesTests(SimpleTestCase):
    def test_existing_choices_keep_auto_and_prefer(self):
        choices = Choices().model_dump()
        self.assertEqual(choices["speculative_method"], "auto")
        self.assertEqual(choices["speculative_decoding"], "prefer")
        self.assertEqual(Choices(speculative_decoding="off").speculative_method, "auto")

    def test_explicit_dspark_cannot_be_disabled_or_use_trt(self):
        for options in ({"speculative_decoding": "off"}, {"frameworks": ["trtllm"]}):
            with self.assertRaises(ValidationError):
                Choices(speculative_method="dspark", **options)
        choices = Choices(
            speculative_method="dspark",
            speculative_decoding="require",
            frameworks=["vllm", "sglang"],
        )
        self.assertEqual(choices.speculative_method, "dspark")

    def test_other_methods_are_not_accepted_as_dspark(self):
        with self.assertRaises(ValidationError):
            Choices(speculative_method="dflash")


@override_settings(
    INFIRO_CREDENTIAL_KEYS=Fernet.generate_key().decode(),
    ALLOWED_HOSTS=["testserver"],
)
class DSparkBoundaryTests(TestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        override = override_settings(INFIRO_PRIVATE_ROOT=Path(root.name))
        override.enable()
        self.addCleanup(override.disable)
        self.user = get_user_model().objects.create_user("dspark-user")
        self.client.force_login(self.user)

    def test_browser_rejects_unmapped_dspark(self):
        rec = Recommendation.objects.create(
            owner=self.user,
            name="Qwen",
            requirements={},
            status="ready",
            selected={"model_id": "org/unmapped", "revision": "a" * 40},
        )
        response = self.client.post(
            "/app/api/prepared",
            json.dumps(
                {
                    "recommendation_id": str(rec.id),
                    "choices": {"speculative_method": "dspark", "speculative_decoding": "require"},
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("No mapped speculative", response.json()["error"])
        self.assertFalse(Prepared.objects.exists())

    def test_editor_cannot_remove_explicit_dspark_even_with_prefer_policy(self):
        source = seed_prepared(self.user)
        source.choices.update(speculative_method="dspark", speculative_decoding="prefer")
        source.payload["server_args"]["vllm"].append("speculative_config=['{}']")
        source.digest = digest(binding(source))
        source.save()
        raw = copy.deepcopy(source.payload)
        raw["server_args"]["vllm"].pop()
        with self.assertRaisesRegex(ValueError, "Required speculative settings"):
            revise_request(self.user, source.id, source.digest, raw)
        raw['server_args'] = {}
        with self.assertRaisesRegex(ValueError, 'Required speculative settings'):
            revise_request(self.user, source.id, source.digest, raw)

    def test_dspark_configuration_error_is_not_retried(self):
        source = seed_prepared(self.user)
        job = Job.objects.get(kind="prepare", target_id=source.id)
        job.attempts = 1
        job.progress["diagnostics"] = {
            "exceptions": [
                {
                    "type": "DSparkConfigurationError",
                    "module": "hf_model_watch.optimization.dspark",
                }
            ]
        }
        job.save()
        fail(job, "DSpark configuration invalid; prepare again.")
        job.refresh_from_db()
        self.assertEqual(job.state, "failed")
