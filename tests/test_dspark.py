import ast
import json
from pathlib import Path

import pytest

from hf_model_watch.optimization.dspark import (
    DRAFTS,
    DSparkConfigurationError,
    capabilities,
    resolve,
)
from hf_model_watch.optimization.preparation import prepare

MODEL = "Qwen/Qwen3-4B"
REV = "a" * 40
DRAFT_REV = "b" * 40
IMAGE = "registry/worker@sha256:" + "c" * 64



@pytest.mark.parametrize("runtime", ["vllm", "sglang"])
@pytest.mark.parametrize("MODEL", list(DRAFTS))
def test_explicit_dspark_reaches_optimizer_request_without_llm_head(runtime, tmp_path, monkeypatch, MODEL):
    monkeypatch.setenv("INFIRO_DSPARK_COMPATIBILITY", "")
    monkeypatch.setattr("hf_model_watch.optimization.dspark.pinned_draft", lambda _: DRAFT_REV)
    run = dict(
        run_id="run_dspark_test",
        selected_model_id=MODEL,
        selection={"revision": REV},
        request_spec=dict(
            speculative_method="dspark",
            speculative_decoding="require",
            gpu_type="gpu-h100-sxm",
            gpu_count=1,
            concurrency=33,
            input_tokens=512,
            output_tokens=128,
            gpu_hourly_cost_usd=1,
            budget_usd_per_hour=5,
        ),
    )
    execution = dict(
        images={runtime: IMAGE},
        bucket="bucket",
        prefix="study",
        preset="1gpu-16vcpu-96gb",
        dry_run=True,
        dspark_compatibility=[],
    )

    # Explicit DSpark must never call research or substitute a head.
    def researcher(*args):
        raise AssertionError("Explicit head must skip research")

    result = prepare(run, {}, execution, tmp_path, Path("config/recommender.yaml"), researcher)
    payload = json.loads((tmp_path / "request.json").read_text())
    assert result["status"] == "prepared"
    assert "dspark_compatibility" not in payload
    assert result["summary"]["speculative_heads"][0]["draft_model"] == DRAFTS[MODEL]
    assert result["summary"]["speculative_method"] == "dspark"
    args = dict(a.split("=", 1) for a in payload["server_args"][runtime])
    if runtime == "vllm":
        import shlex

        config = json.loads(shlex.split(ast.literal_eval(args["speculative_config"])[0])[0])
        assert config == {
            "method": "dspark",
            "model": DRAFTS[MODEL],
            "revision": DRAFT_REV,
            "num_speculative_tokens": 7,
        }
        from llm_optimizer import args as optimizer_args
        from llm_optimizer import predefined
        from llm_optimizer.remote.worker import optimizer_command

        command = optimizer_command(
            {"request": payload, "framework": runtime}, tmp_path / "out.json"
        )
        value = next(
            command[i + 1]
            for i, arg in enumerate(command)
            if arg == "--server-args" and command[i + 1].startswith("speculative_config=")
        )
        sets = optimizer_args.parse_args_str(
            value, optimizer_args.ArgScope.SERVER, predefined.SERVER_CONFIGS[runtime], False
        )
        native = sets[0].get_all_possible_arg_values()[0]
        rendered = predefined.get_server_cmd_args(runtime, [native])
        argv = shlex.split("vllm serve " + MODEL + " " + " ".join(rendered))
        assert (
            json.loads(
                next(a.split("=", 1)[1] for a in argv if a.startswith("--speculative-config="))
            )
            == config
        )
    else:
        assert ast.literal_eval(args["speculative_algorithm"]) == ["DSPARK"]
        assert ast.literal_eval(args["speculative_num_draft_tokens"]) == [8]
        assert ast.literal_eval(args["speculative_draft_model_path"]) == [DRAFTS[MODEL]]
        from llm_optimizer.remote.worker import optimizer_command
        from llm_optimizer import args as optimizer_args, predefined
        command = optimizer_command({"request": payload, "framework": runtime}, tmp_path / "out.json")
        for i, arg in enumerate(command):
            if arg == '--server-args' and command[i + 1].startswith('speculative_'):
                sets = optimizer_args.parse_args_str(command[i + 1], optimizer_args.ArgScope.SERVER,
                                                     predefined.SERVER_CONFIGS[runtime], False)
                assert sets[0].get_all_possible_arg_values()
        assert "speculative_draft_model_revision=" + repr([DRAFT_REV]) in command



def test_unsupported_model_and_framework():
    with pytest.raises(DSparkConfigurationError, match="target model"):
        resolve("google/gemma-4-12B-it", REV, ["vllm"], {"vllm": IMAGE})
    with pytest.raises(DSparkConfigurationError, match="TensorRT"):
        resolve(MODEL, REV, ["trtllm"], {"trtllm": IMAGE})
    with pytest.raises(DSparkConfigurationError, match="immutable"):
        resolve(MODEL, REV, ["vllm"], {"vllm": "mutable:latest"})
    assert set(capabilities({"vllm": IMAGE})["drafts"]) == set(DRAFTS)


@pytest.mark.parametrize("runtime", ["vllm", "sglang"])
def test_remote_settings_match_optimizer_notebook_helper(runtime, tmp_path, monkeypatch):
    """Check parity against the real helper with CPU-only setup doubles.

    Mocked CUDA/CLI/snapshots are configuration tests, not runtime verification.
    """
    import importlib.metadata
    import subprocess
    import sys
    from types import SimpleNamespace

    import huggingface_hub
    from llm_optimizer.experiment.hf_models import configure_hf_models

    target = tmp_path / "target" / REV
    draft = tmp_path / "draft" / DRAFT_REV
    for path in (target, draft):
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"block_size": 7}))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda _: "CPU-test-double"
            ),
            version=SimpleNamespace(cuda="test"),
        ),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **_: str(draft if repo_id == DRAFTS[MODEL] else target),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0,
            stdout="DSPARK --speculative-algorithm --speculative-draft-model-path "
            "--speculative-num-draft-tokens --speculative-config",
            stderr="",
        ),
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "test-driver")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.speculative.spec_info",
        SimpleNamespace(
            SpeculativeAlgorithm=SimpleNamespace(from_string=lambda _: "DSPARK"),
        ),
    )
    context = dict(
        RUN_REAL_GPU=True,
        REAL_FRAMEWORKS=[runtime],
        REAL_PREFLIGHT={("B", runtime): {"fixed_server": {}, "expanded_choices": {}}},
        MODEL_SNAPSHOTS={},
        PHASE_MODELS={},
        RUN_DIR=tmp_path,
        expanded_choices=lambda *a: None,
    )
    options = (
        {"enable_dflash": True, "num_speculative_tokens": 7}
        if runtime == "vllm"
        else {
            "speculative_algorithm": "DSPARK",
            "block_size": 7,
        }
    )
    configure_hf_models(
        context,
        MODEL,
        REV,
        framework=runtime,
        draft_model_id=DRAFTS[MODEL],
        draft_revision=DRAFT_REV,
        **options,
    )
    reference = context["REAL_PREFLIGHT"][("B", runtime)]["fixed_server"]
    arguments, _ = resolve(MODEL, REV, [runtime], {runtime: IMAGE}, DRAFT_REV)
    resolved = {
        key: ast.literal_eval(value)[0]
        for arg in arguments[runtime]
        for key, value in [arg.split("=", 1)]
    }
    if runtime == "sglang":
        assert resolved.pop("speculative_draft_model_revision") == DRAFT_REV
        reference["speculative_draft_model_path"] = DRAFTS[MODEL]
        assert resolved == reference
    else:
        config = json.loads(resolved["speculative_config"])
        assert config["method"] == reference["speculative_config"]["method"]
        assert (
            config["num_speculative_tokens"]
            == reference["speculative_config"]["num_speculative_tokens"]
        )
        assert config["revision"] == DRAFT_REV

@pytest.mark.parametrize('block_size', [7, 8])
def test_pinned_head_configuration_check(tmp_path, monkeypatch, block_size):
    import huggingface_hub
    from types import SimpleNamespace
    from hf_model_watch.optimization.dspark import pinned_draft
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'block_size': block_size}))
    monkeypatch.setattr(huggingface_hub.HfApi, 'model_info', lambda _, repo: SimpleNamespace(sha=DRAFT_REV))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda repo, filename, revision: str(config))
    if block_size == 7:
        assert pinned_draft(MODEL) == DRAFT_REV
    else:
        with pytest.raises(DSparkConfigurationError, match='block_size'):
            pinned_draft(MODEL)
