import builtins
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from hf_model_watch.optimization.objectives import objective_options

from .contracts import Choices


class ObjectiveChoicesTests(SimpleTestCase):
    def test_all_pairs_and_triple_work_at_every_concurrency(self):
        for concurrency in (1, 24, 128):
            for option in objective_options():
                names = option["value"].split(",")
                with self.subTest(concurrency=concurrency, names=names):
                    self.assertEqual(Choices(concurrency=concurrency,
                        objectives=names).objectives, names)
                    if len(names) > 1:
                        with self.assertRaisesRegex(ValueError, "Pruning requires"):
                            Choices(concurrency=concurrency, objectives=names,
                                    pruner="MedianPruner")

    def test_new_web_defaults_use_p95_and_saved_mean_objectives_stay_mean(self):
        self.assertEqual(Choices().objectives, ["p95_e2e_latency_ms", "p95_ttft_ms"])
        self.assertEqual(Choices(concurrency=24).objectives,
                         ["output_throughput", "p95_e2e_latency_ms"])
        self.assertEqual(Choices(objectives=["mean_ttft_ms"]).objectives, ["mean_ttft_ms"])


class BootstrapWithoutOptimizerTests(TestCase):
    def test_workspace_loads_without_optimizer_in_web_environment(self):
        self.client.force_login(User.objects.create_user(username="metadata-only"))
        original_import = builtins.__import__

        def without_optimizer(name, *args, **kwargs):
            if name == "llm_optimizer" or name.startswith("llm_optimizer."):
                raise ModuleNotFoundError("No module named 'llm_optimizer'", name=name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_optimizer):
            response = self.client.get("/app/api/bootstrap")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["optimization_objectives"]), 7)
