from __future__ import annotations

from hf_model_watch.errors import FailureCategory
from hf_model_watch.runtime.runtime_resolver import resolve_runtime
from hf_model_watch.runtime.smoke_tests import run_local_smoke_test
from hf_model_watch.storage.sqlite_store import SQLiteStore


def test_runtime_resolver_chooses_transformers_when_vllm_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "hf_model_watch.runtime.runtime_resolver.vllm_probe",
        lambda metadata: {
            "available": False,
            "architecture_appears_supported": True,
            "usable": False,
            "message": "vLLM is not installed",
        },
    )

    decision = resolve_runtime(_metadata(), _scan_config(), _serverless_config())

    assert decision.fail_fast is False
    assert decision.backend == "transformers"
    assert decision.dtype == "bf16"


def test_runtime_resolver_blocks_trust_remote_code_when_disallowed() -> None:
    metadata = _metadata()
    metadata["config"] = {"auto_map": {"AutoModelForCausalLM": "custom.Model"}}

    decision = resolve_runtime(metadata, _scan_config(), _serverless_config())

    assert decision.fail_fast is True
    assert decision.failure_category == FailureCategory.REQUIRES_TRUST_REMOTE_CODE.value


def test_tiny_model_smoke_path_succeeds_with_mocked_backend(tmp_path, monkeypatch) -> None:
    class FakeGeneration:
        def as_dict(self) -> dict[str, object]:
            return {
                "prompt": "Hello, my name is",
                "text": "Hello, my name is Test",
                "prompt_tokens": 5,
                "generated_tokens": 2,
                "elapsed_seconds": 0.01,
            }

    class FakeConfig:
        storage = {"local_artifact_dir": str(tmp_path / "artifacts")}
        scan = _ModelDump(_scan_config())
        serverless = _ModelDump(_serverless_config())

        def model_dump(self) -> dict[str, object]:
            return {
                "storage": self.storage,
                "scan": self.scan.model_dump(),
                "serverless": self.serverless.model_dump(),
            }

    monkeypatch.setattr(
        "hf_model_watch.runtime.smoke_tests.collect_metadata_claims",
        lambda model_id: {
            **_metadata(),
            "model_id": model_id,
            "revision": "abc123",
        },
    )
    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()

    result = run_local_smoke_test(
        run_id="run_test",
        model_id="sshleifer/tiny-gpt2",
        config=FakeConfig(),
        store=store,
        generate_fn=lambda model_id, revision, load_args: FakeGeneration(),
        eval_generate_fn=lambda prompt, max_new_tokens: FakeGeneration(),
    )

    run_dir = tmp_path / "artifacts" / "run_test"
    assert result["observed"]["tokenizer_load_success"] is True
    assert (run_dir / "metadata" / "claims_vs_observed.json").exists()
    assert (run_dir / "raw_generations.jsonl").exists()
    assert (run_dir / "runtime_result.json").exists()
    assert store.already_tested_revision("sshleifer/tiny-gpt2", "abc123") is True


class _ModelDump:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, object]:
        return self._data


def _metadata() -> dict[str, object]:
    return {
        "model_id": "sshleifer/tiny-gpt2",
        "revision": "abc123",
        "tags": ["text-generation", "license:mit"],
        "license": "mit",
        "gated": False,
        "has_safetensors": True,
        "architecture": ["GPT2LMHeadModel"],
        "config": {"architectures": ["GPT2LMHeadModel"], "max_position_embeddings": 1024},
        "context_length": 1024,
        "estimated_params_billion": 0.1,
    }


def _scan_config() -> dict[str, object]:
    return {
        "allow_trust_remote_code": False,
        "max_params_billion": 30,
    }


def _serverless_config() -> dict[str, object]:
    return {
        "gpu_type": "H100",
        "gpu_memory_gb": 80,
        "memory_safety_margin": 0.85,
        "max_params_billion": 30,
        "max_download_gb": 150,
    }
