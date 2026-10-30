import copy
import json
import shlex

import pytest

from hf_model_watch.optimization.contracts import NewStudyIntent, StudyContext, Workload
from hf_model_watch.optimization.runtime_constraints import GEMMA, validate_request
from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter

REV = "a" * 40


def build(model=GEMMA, concurrency=33, **execution):
    context = StudyContext(
        model_id=model,
        revision=REV,
        gpu_type="gpu-h100-sxm",
        gpu_count=1,
        concurrency=concurrency,
        workload=Workload(input_tokens=512, output_tokens=128),
    )
    intent = NewStudyIntent(
        context=context, frameworks=["vllm", "sglang", "trtllm"], idempotency_key="gemma-test"
    )
    return OptimizationAdapter(None, {"bucket": "test", **execution}).build_request(intent)


def test_final_domains_and_native_commands(tmp_path):
    from llm_optimizer import args as native
    from llm_optimizer import predefined
    from llm_optimizer.predefined.trtllm import load_native_config
    from llm_optimizer.remote.worker import optimizer_command
    from llm_optimizer.search_profiles import EXPANDED_CHOICES, SPACES

    original = copy.deepcopy((EXPANDED_CHOICES, SPACES))
    request = build()
    validate_request(request)
    assert "max_concurrency=33" in request["client_args"]
    assert "server_ready_timeout_seconds" not in request
    from llm_optimizer.remote.contracts import OptimizationRequest

    assert OptimizationRequest.parse(request).timeout_minutes == 180
    assert "cache_pinned_model_assets" not in request
    for framework in request["frameworks"]:
        command = optimizer_command(
            {"request": request, "framework": framework}, tmp_path / "out.json"
        )
        rendered = []
        for i, arg in enumerate(command):
            if arg != "--server-args":
                continue
            sets = native.parse_args_str(
                command[i + 1], native.ArgScope.SERVER, predefined.SERVER_CONFIGS[framework], False
            )
            for domain in sets:
                values = domain.get_all_possible_arg_values()
                rendered.extend(predefined.get_server_cmd_args(framework, [values[0]]))
        argv = shlex.split(" ".join(rendered))
        if framework == "vllm":
            assert "--attention-backend=auto" in argv
            assert "--kv-cache-dtype=auto" in argv
        elif framework == "sglang":
            assert "--attention-backend=triton" in argv
            assert "--max-running-requests=8" in argv
            assert "--context-length=1024" in argv
        else:
            path = next(v.split("=", 1)[1] for v in argv if v.startswith("--config="))
            config = load_native_config({"config": path})
            assert config["attn_backend"] == "FLASHINFER"
            assert config["enable_chunked_prefill"] is True
            assert "--attn_backend=" not in " ".join(argv)
    assert (EXPANDED_CHOICES, SPACES) == original
    other = build("Qwen/Qwen3-4B")
    assert "attention_backend=['auto', 'FLASH_ATTN', 'FLASHINFER']" in other["server_args"]["vllm"]
    assert other.get("server_ready_timeout_seconds", {}) == {}
    assert all(
        json.loads(text)["attn_backend"] == "TRTLLM"
        for text in other["server_config_files"]["trtllm"].values()
    )


@pytest.mark.parametrize(
    "framework,arg",
    [
        ("vllm", "attention_backend=['FLASH_ATTN']"),
        ("vllm", "kv_cache_dtype=['fp8']"),
        ("sglang", "attention_backend=['flashinfer']"),
        ("sglang", "max_running_requests=[96]"),
        ("sglang", "cuda_graph_backend_prefill=['breakable']"),
    ],
)
def test_overrides_cannot_restore_prohibited_domains(framework, arg):
    from llm_optimizer.search_profiles import merge_server_args

    request = build()
    request["server_args"][framework] = merge_server_args(request["server_args"][framework], [arg])
    with pytest.raises(ValueError, match="Gemma runtime constraint"):
        validate_request(request)
    with pytest.raises(ValueError, match="Gemma runtime constraint"):
        build(optuna_by_framework={framework: {"server_args": [arg]}})


def test_missing_constraints_and_native_config_bypass_rejected():
    request = build()
    request["server_args"]["vllm"] = []
    with pytest.raises(ValueError, match="Gemma runtime constraint"):
        validate_request(request)
    request = build()
    for name in request["server_config_files"]["trtllm"]:
        request["server_config_files"]["trtllm"][name] = (
            '{"attn_backend":"TRTLLM","enable_chunked_prefill":true}'
        )
    with pytest.raises(ValueError, match="FLASHINFER"):
        validate_request(request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("server_ready_timeout_seconds", {"vllm": 900}),
        ("cache_pinned_model_assets", True),
    ],
)
def test_deferred_worker_fields_rejected(field, value):
    with pytest.raises(ValueError, match="Worker extensions are deferred"):
        build(**{field: value})


def test_shipped_optimizer_uses_original_worker_contract():
    from llm_optimizer.remote.contracts import OptimizationRequest

    assert "server_ready_timeout_seconds" not in OptimizationRequest.__dataclass_fields__
    assert "cache_pinned_model_assets" not in OptimizationRequest.__dataclass_fields__
    payload = OptimizationRequest.parse(build()).to_dict()
    assert "server_ready_timeout_seconds" not in payload
    assert "cache_pinned_model_assets" not in payload


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
def test_gemma_preparation_retains_research_and_workload(framework, tmp_path):
    from hf_model_watch.optimization.preparation import prepare

    calls = []

    def researcher(model, revision, *args):
        calls.append((model, revision))
        return {
            "model": model,
            "revision": revision,
            "sources": [],
            "inspection": {"claims": [], "heads": []},
        }

    run = {
        "run_id": "run_gemma_prepare",
        "selected_model_id": GEMMA,
        "selection": {"revision": REV},
        "request_spec": {
            "frameworks": [framework],
            "gpu_type": "gpu-h100-sxm",
            "gpu_count": 1,
            "input_tokens": 512,
            "output_tokens": 128,
            "concurrency": 33,
            "gpu_hourly_cost_usd": 1,
            "budget_usd_per_hour": 5,
        },
    }
    execution = {
        "images": {framework: "registry/worker@sha256:" + "b" * 64},
        "bucket": "test",
        "preset": "1gpu-16vcpu-200gb",
        "dry_run": True,
    }
    result = prepare(run, {}, execution, tmp_path, tmp_path, researcher)
    assert result["status"] == "prepared"
    request = json.loads((tmp_path / "request.json").read_text())
    validate_request(request)
    assert calls == [(GEMMA, REV)]
    assert "max_concurrency=33" in request["client_args"]
    assert request["semantic_context"]["workload"]["input_tokens"] == 512
    assert result["summary"]["speculative_heads"] == []
