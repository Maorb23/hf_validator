"""CPU-only tests of the shared notebook/controller dataset workload contract."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_optimizer.hf_workload import (
    dataset_id,
    first_human,
    prepare_hf_workload,
    validate_manifest,
)
from hf_model_watch.optimization.catalog import incompatibilities
from hf_model_watch.optimization.contracts import NewStudyIntent, StudyContext
from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter

REV = "a" * 40


class Tokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def materialize(**overrides):
    rows = [
        {
            "conversations": [
                {"from": "system", "value": "rules"},
                {"from": "human", "value": str(i)},
                {"from": "gpt", "value": "answer"},
            ]
        }
        for i in range(10)
    ]
    args = dict(
        dataset={"repo_id": "org/data"},
        model="org/model",
        model_revision=REV,
        input_tokens=8,
        output_tokens=4,
        num_prompts=3,
        seed=12,
        api=SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha=REV)),
        loader=lambda *a, **kw: rows,
        tokenizer=Tokenizer(),
    )
    args.update(overrides)
    return prepare_hf_workload(**args)


def context(dataset=None):
    return StudyContext(
        model_id="org/model",
        revision=REV,
        gpu_type="gpu-h100-sxm",
        concurrency=1,
        workload={
            "input_tokens": 8,
            "output_tokens": 4,
            "num_prompts": 3,
            "seed": 12,
            "dataset": dataset,
        },
    )


def test_reproducible_fixed_workload():
    a, b = materialize(), materialize()
    assert a == b
    assert a["sha256"] != materialize(seed=13)["sha256"]
    assert len(set(a["row_indices"])) == 3
    assert all(len(r["prompt"]) == 8 and r["output_len"] == 4 for r in a["requests"])
    assert a["dataset"]["revision"] == REV
    assert a["tokenizer_revision"] == REV
    a["requests"][0]["prompt"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(a)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"conversations": []},
        {"conversations": [{"from": "user", "value": "hello"}]},
        {"conversations": [{"from": "human", "value": " "}]},
        {"conversations": [{"from": "gpt", "value": "answer"}]},
    ],
)
def test_invalid_schema(row):
    with pytest.raises(ValueError):
        materialize(loader=lambda *a, **kw: [row])


def test_first_human_not_first_turn():
    assert (
        first_human(
            {
                "conversations": [
                    {"from": "system", "value": "rules"},
                    {"from": "human", "value": "prompt"},
                    {"from": "human", "value": "later"},
                ]
            }
        )
        == "prompt"
    )


def test_insufficient_rows_never_repeat_records():
    with pytest.raises(ValueError, match="Insufficient"):
        materialize(num_prompts=11)


def test_missing_access_redacts_error_no_fallback():
    loader = Mock(side_effect=RuntimeError("secret-hf-token"))
    with pytest.raises(ValueError, match="loading or access failed") as error:
        materialize(loader=loader)
    assert "secret-hf-token" not in str(error.value)
    assert loader.call_count == 1


def test_revision_pinned_before_load():
    loader = Mock(return_value=[{"conversations": [{"from": "human", "value": "x"}]}])
    materialize(loader=loader, num_prompts=1)
    assert loader.call_args.kwargs["revision"] == REV


@pytest.mark.parametrize(
    "value",
    [
        "https://huggingface.co/datasets/org/data?token=secret",
        "https://user:secret@huggingface.co/datasets/org/data",
        "/tmp/data",
    ],
)
def test_credentials_and_local_paths_rejected(value):
    with pytest.raises(ValueError):
        dataset_id(value)


def test_adapter_pins_and_reuses_manifest(monkeypatch, tmp_path):
    import llm_optimizer.hf_workload as shared

    loader = Mock(return_value=materialize())
    monkeypatch.setattr(shared, "prepare_hf_workload", loader)
    intent = NewStudyIntent(
        context=context({"repo_id": "https://huggingface.co/datasets/org/data"}),
        frameworks=["vllm", "sglang"],
        idempotency_key="dataset-test",
    )
    adapter = OptimizationAdapter(
        None, {"bucket": "test", "images": {"vllm": "image", "sglang": "image"}}
    )
    payload = adapter.build_request(intent)
    assert payload == adapter.build_request(intent)
    assert loader.call_count == 1
    assert intent.context.workload.dataset.revision == REV
    assert intent.context.workload.workload_sha256 == payload["workload_manifest"]["sha256"]
    assert incompatibilities(context(), intent.context) == ["workload"]
    from llm_optimizer.remote.contracts import OptimizationRequest
    from llm_optimizer.remote.worker import optimizer_command

    parsed = OptimizationRequest.parse(payload)
    assert parsed.to_dict()["workload_manifest"] == materialize()
    for framework in intent.frameworks:
        cmd = optimizer_command(
            {"request": payload, "framework": framework}, tmp_path / "results.json"
        )
        assert any(x.startswith("workload_path=") for x in cmd)
    altered = copy.deepcopy(payload)
    altered["workload_manifest"]["requests"][0]["prompt"] = "changed"
    with pytest.raises(ValueError):
        OptimizationRequest.parse(altered)


def test_default_random_workload_unchanged():
    intent = NewStudyIntent(context=context(), frameworks=["vllm"], idempotency_key="default")
    payload = OptimizationAdapter(None, {}).build_request(intent)
    assert "workload_manifest" not in payload
    assert payload["client_args"] == [
        "dataset_name=random",
        "random_input_len=8",
        "random_output_len=4",
        "random_range_ratio=1.0",
        "num_prompts=3",
        "seed=12",
        "max_concurrency=1",
    ]


def test_bad_custom_path_does_not_download(monkeypatch):
    from llm_optimizer import bench_client

    download = Mock(side_effect=AssertionError("fallback attempted"))
    monkeypatch.setattr(bench_client, "download_and_cache_file", download)
    with pytest.raises(ValueError, match="Custom dataset_path"):
        bench_client.sample_random_requests(8, 4, 3, 1.0, Tokenizer(), "/missing/custom.json")
    download.assert_not_called()


def test_manifest_path_skips_tokenizer_and_dataset(monkeypatch, tmp_path):
    from llm_optimizer import bench_client

    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(materialize()))
    monkeypatch.setattr(bench_client, "get_tokenizer", Mock(side_effect=AssertionError("reloaded")))
    manifest, metadata = bench_client.prepare_workload_manifest(
        {"model": "org/model", "workload_path": str(path)}
    )
    assert manifest == materialize()
    assert metadata["workload_sha256"] == manifest["sha256"]


def test_controller_stages_identical_workload_for_frameworks(tmp_path):
    from llm_optimizer.remote.controller import Controller

    provider = Mock()
    provider.submit.side_effect = ["aijob-1", "aijob-2"]
    store = Mock()
    uploaded = []
    store.upload.side_effect = lambda path, run_id, framework: uploaded.append(
        (json.loads(path.read_text()), run_id, framework)
    )
    payload = {
        "model": "org/model",
        "revision": REV,
        "frameworks": ["vllm", "sglang"],
        "gpu_type": "gpu-h100-sxm",
        "images": {"vllm": "image", "sglang": "image"},
        "bucket": "test",
        "dry_run": False,
        "workload_manifest": materialize(),
    }
    controller = Controller(tmp_path / "controller.db", provider, lambda *a: store, concurrency=2)
    submitted = controller.submit(payload)
    controller.tick()
    assert len(uploaded) == 2
    assert all(item == (materialize(), submitted["run_id"], "workload") for item in uploaded)
    assert all(
        call.args[0]["request"]["workload_manifest"] == materialize()
        for call in provider.submit.call_args_list
    )


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_worker_rejects_unavailable_or_corrupt_staged_workload(tmp_path, mode):
    from llm_optimizer.remote.worker import run_worker

    store = Mock()

    def download(path, *args):
        if mode == "missing":
            return False
        value = materialize()
        value["requests"][0]["prompt"] = "corrupt"
        path.write_text(json.dumps(value))
        return True

    store.download.side_effect = download
    runner = Mock(side_effect=AssertionError("must not launch"))
    payload = {
        "model": "org/model",
        "frameworks": ["vllm"],
        "gpu_type": "gpu-h100-sxm",
        "images": {"vllm": "image"},
        "bucket": "test",
        "dry_run": False,
        "workload_manifest": materialize(),
    }
    with pytest.raises(ValueError):
        run_worker(
            {"run_id": "run", "framework": "vllm", "request": payload},
            store=store,
            work_dir=tmp_path,
            runner=runner,
        )
    runner.assert_not_called()


def test_native_optimizer_hf_arguments_use_shared_preparer(monkeypatch):
    from llm_optimizer import bench_client
    from llm_optimizer.predefined.client import CLIENT_CONFIGS
    import llm_optimizer.hf_workload as shared

    prepare = Mock(return_value=materialize())
    monkeypatch.setattr(shared, "prepare_hf_workload", prepare)
    args = {
        "model": "org/model",
        "model_revision": REV,
        "hf_dataset": "org/data",
        "hf_dataset_revision": "branch",
        "hf_dataset_config": "custom",
        "hf_dataset_split": "test",
        "num_prompts": 3,
        "seed": 12,
        "random_input_len": 8,
        "random_output_len": 4,
    }
    manifest, metadata = bench_client.prepare_workload_manifest(args)
    assert manifest == materialize()
    assert metadata["tokenizer_revision"] == REV
    assert metadata["dataset_identity"]["revision"] == REV
    assert metadata["row_indices"] == manifest["row_indices"]
    assert prepare.call_args.args[0] == {
        "repo_id": "org/data",
        "revision": "branch",
        "config": "custom",
        "split": "test",
    }
    assert all(
        key in CLIENT_CONFIGS
        for key in (
            "hf_dataset",
            "hf_dataset_revision",
            "hf_dataset_config",
            "hf_dataset_split",
            "workload_path",
        )
    )


def test_preparation_contract_rejects_credential_fields():
    from hf_model_watch.optimization.preparation import PreparationInputs

    with pytest.raises(ValueError):
        PreparationInputs(run_id="run", dataset={"repo_id": "org/data", "token": "secret"})


def test_hf_dataset_access_failure_never_calls_loader():
    loader = Mock()
    api = Mock()
    api.dataset_info.side_effect = RuntimeError("token=secret")
    with pytest.raises(ValueError, match="loading or access failed") as exc:
        materialize(api=api, loader=loader)
    assert "secret" not in str(exc.value)
    loader.assert_not_called()


def test_remote_raw_hf_source_cannot_reload_per_framework():
    from llm_optimizer.remote.contracts import OptimizationRequest
    with pytest.raises(ValueError, match='prepared once'):
        OptimizationRequest.parse({'model': 'org/model', 'frameworks': ['vllm'],
            'gpu_type': 'gpu-h100-sxm', 'images': {'vllm': 'image'}, 'bucket': 'test',
            'client_args': ['hf_dataset=org/data']})


def test_default_remote_payload_works_with_older_worker_contract():
    from llm_optimizer.remote.contracts import OptimizationRequest
    request = OptimizationRequest.parse({'model': 'org/model', 'frameworks': ['vllm'],
        'gpu_type': 'gpu-h100-sxm', 'images': {'vllm': 'image'}, 'bucket': 'test'})
    assert 'workload_manifest' not in request.to_dict()
