"""Focused publication-boundary tests; no runtime or provider dependencies."""
import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("public_export", Path(__file__).resolve().parents[2] / "scripts/export_public_inference_results.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {"metadata": {"model_revision": "a"*40, "measurement": "MEASURED", "gpu_type": "test-observed-gpu", "gpu_count": 1, "framework_version": "1.0", "target_model": "example/model", "requests_sha256": "b"*64}, "config": {"framework": "vllm", "client_args": {"random_input_len": 128, "random_output_len": 32, "num_prompts": 100, "max_concurrency": 8, "seed": 11}, "server_args": {"max_num_seqs": 32, "private_path": "/private/model"}}, "results": {"benchmark_valid": True, "completed": 100, "failed": 0, "mean_e2e_latency_ms": 100, "mean_ttft_ms": 20, "output_throughput": 1000}}


class ExportTests(unittest.TestCase):
    def test_sanitizes_and_preserves_inference_context(self):
        result = module.normalize(fixture(), "c"*64, 1)
        self.assertEqual(result["configuration"], {"max_num_seqs": 32})
        self.assertEqual(result["context"]["revision"], "a"*40)
        self.assertNotIn("quality", result)

    def test_rejects_placeholder_revision(self):
        data = fixture(); data["metadata"]["model_revision"] = "REPLACE_WITH_COMMIT"
        with self.assertRaisesRegex(ValueError, "immutable"):
            module.normalize(data, "c"*64, 1)

    def test_rejects_partial_and_nonfinite_measurements(self):
        for key, value in [("completed", 99), ("output_throughput", math.nan), ("mean_ttft_ms", 0)]:
            data = fixture(); data["results"][key] = value
            with self.assertRaises(ValueError):
                module.normalize(data, "c"*64, 1)


if __name__ == "__main__":
    unittest.main()
