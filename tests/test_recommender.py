from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.error import HTTPError

import httpx
import pytest
from openai import NotFoundError
from typer.testing import CliRunner

from hf_model_watch.cli import app
from hf_model_watch.orchestration.job_payloads import WorkerJobPayload
from hf_model_watch.recommender import core
from hf_model_watch.recommender.benchmark import benchmark_recommendation
from hf_model_watch.recommender.benchmark_run import run_fixture_benchmark
from hf_model_watch.recommender.core import RequestSpec, load_settings, rank_candidates, recommend
from hf_model_watch.recommender.report import build_benchmark_report
from hf_model_watch.recommender.server_runtime import build_server_command
from hf_model_watch.runtime.backend_transformers import GenerationResult
from hf_model_watch.storage.sqlite_store import SQLiteStore


def _mock_nebius(monkeypatch, respond):
    def create(**payload):
        message = respond(payload)
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {"choices": [{"message": message}]},
            choices=[SimpleNamespace(
                message=SimpleNamespace(model_dump=lambda **_kwargs: message),
                finish_reason="stop",
            )]
        )

    monkeypatch.setattr(
        core,
        "OpenAI",
        lambda **_kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )


def _spec(**updates):
    data = dict(
        request="Score short posts",
        gpu_memory_gb=80,
        gpu_count=1,
        budget_usd_per_hour=5,
        gpu_hourly_cost_usd=3,
        license="apache-2.0",
        framework_mode="fast",
    )
    return RequestSpec(**(data | updates))


def _record(name, params):
    return dict(
        model_id=name,
        revision="abc",
        license="apache-2.0",
        estimated_params_billion=params,
        has_weights=True,
        has_config=True,
        config={"architectures": ["ExampleForCausalLM"]},
        context_length=131072,
        context_length_source="metadata",
    )


def test_smallest_candidate_wins_and_constraints_block():
    settings = load_settings()
    assert settings["nebius"]["default_model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert "us-central1" in settings["nebius"]["base_url"]
    rows = rank_candidates(
        [_record("org/large", 7), _record("org/small", 3)], _spec(), {}, settings
    )
    assert rows[0]["model_id"] == "org/small"
    assert rows[0]["runtime"] == "transformers"
    assert not rank_candidates([_record("org/x", 3)], _spec(license="mit"), {}, settings)[0][
        "eligible"
    ]
    assert not rank_candidates([_record("org/x", 40)], _spec(gpu_memory_gb=16), {}, settings)[0][
        "eligible"
    ]
    assert not rank_candidates(
        [_record("org/x", 3)], _spec(gpu_hourly_cost_usd=None), {}, settings
    )[0]["eligible"]
    assert rank_candidates([_record("org/x", 3)], _spec(license="irrelevant"), {}, settings)[0][
        "eligible"
    ]
    conflict = {
        "org/x": [
            {
                "source_url": "https://example.org",
                "origin": "tavily",
                "retrieved_text": "This model is licensed under MIT.",
            }
        ]
    }
    assert (
        "conflicting_license_claims"
        in rank_candidates([_record("org/x", 3)], _spec(), conflict, settings)[0][
            "blocking_reasons"
        ]
    )


def test_agent_tool_budget_and_saved_artifact(tmp_path):
    settings = load_settings()
    settings["nebius"]["max_tool_calls"] = 1
    responses = iter(
        [
            {
                "tool_calls": [
                    {
                        "id": "one",
                        "function": {"name": "hf_lookup", "arguments": '{"model_id":"org/x"}'},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "two",
                        "function": {"name": "hf_lookup", "arguments": '{"model_id":"org/y"}'},
                    }
                ]
            },
        ]
    )
    result = recommend(
        _spec(),
        settings,
        tmp_path,
        chat=lambda: next(responses),
        lookup=lambda **kwargs: [_record("org/x", 3)],
        search=lambda *_args, **_kwargs: [],
    )
    assert result["status"] == "insufficient_evidence"
    assert any(event["event"] == "tool_limit_reached" for event in result["events"])
    assert (
        json.loads((tmp_path / "recommendation.json").read_text())["status"]
        == "insufficient_evidence"
    )
    prevalidation = json.loads(
        (tmp_path / "nemotron_prevalidation_response.json").read_text()
    )
    assert len(prevalidation["messages"]) == 2
    assert prevalidation["messages"][0]["tool_calls"][0]["function"]["name"] == "hf_lookup"
    assert prevalidation["final_message"] is None


def test_cli_needs_input_json():
    result = CliRunner().invoke(app, ["recommend", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "needs_input"
    assert "framework_mode" in data["needs_input"]


def test_nebius_404_identifies_configuration_without_exposing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "secret-key")

    def missing_model(_payload):
        request = httpx.Request("POST", "https://example.org/chat/completions")
        response = httpx.Response(404, request=request)
        raise NotFoundError("Not Found", response=response, body=None)

    _mock_nebius(monkeypatch, missing_model)
    with pytest.raises(ValueError, match="Nebius endpoint or model not found") as exc:
        recommend(_spec(), load_settings(), tmp_path, search=lambda *_args, **_kwargs: [])
    assert "secret-key" not in str(exc.value)


def test_benchmark_requires_confirmation(tmp_path):
    path = tmp_path / "recommendation.json"
    item = {
        "model_id": "org/x",
        "revision": "abc",
        "license": "apache-2.0",
        "blocking_reasons": [],
        "compatible_frameworks": ["transformers"],
        "estimated_bf16_memory_gb": 8,
    }
    path.write_text(
        json.dumps(
            {
                "status": "recommended",
                "recommendation": item,
                "framework_mode": "fast",
                "evaluated_candidates": [item],
                "benchmark_plan": [
                    {
                        "model_id": "org/x",
                        "revision": "abc",
                        "task_fixture": "social_scoring",
                        "frameworks": ["transformers"],
                    }
                ],
            }
        )
    )
    result = benchmark_recommendation(path, submit=lambda *_args, **_kwargs: 1)
    assert result["status"] == "awaiting_confirmation"
    assert result["submitted_jobs"] == []


def test_hf_card_is_saved_and_missing_fields_explicit(tmp_path, monkeypatch):
    card = tmp_path / "README.md"
    card.write_text("# Model card\n" + "capability " * 900)
    monkeypatch.setattr(
        core, "lookup_models", lambda **_kwargs: [_record("org/x", 3) | {"license": None}]
    )
    monkeypatch.setattr(core, "hf_hub_download", lambda *_args, **_kwargs: str(card))
    rows = core.hf_lookup(model_id="org/x", artifact_dir=tmp_path / "evidence")
    assert rows[0]["card_artifact"]
    assert len(rows[0]["card_excerpt"]) == 4000
    assert "license" in rows[0]["missing_fields"]


def test_tavily_fallback_keeps_provenance_and_cache(tmp_path, monkeypatch):
    settings = load_settings()
    settings["tavily"]["cache_dir"] = str(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    calls = []

    def fake_post(_url, payload, _headers, _timeout):
        calls.append(payload)
        return {
            "results": [
                {"url": "https://example.org/review", "content": "Support claim remains uncertain"}
            ]
        }

    monkeypatch.setattr(core, "_post_json", fake_post)
    first = core.tavily_research("org/x license", settings)
    second = core.tavily_research("org/x license", settings)
    assert first == second
    assert first[0]["origin"] == "tavily"
    assert first[0]["confidence"] == "uncertain"
    assert len(calls) == 1
    assert calls[0]["search_depth"] == "advanced"
    assert calls[0]["chunks_per_source"] == 3
    assert calls[0]["include_raw_content"] == "markdown"
    assert calls[0]["max_results"] == 8
    assert "secret" not in next(tmp_path.iterdir()).read_text()


def test_research_queries_target_missing_fields_and_speculative_evidence():
    record = _record("org/x", 3) | {
        "tokenizer_config": {"chat_template": "{{ messages }}"},
        "estimated_download_size_gb": 6,
    }
    cases = dict(core.model_research_queries(record, "Use Medusa heads"))
    assert "speculative" in cases
    assert "origin" in cases
    assert "chat" not in cases
    assert "params" not in cases
    assert "license" not in cases


def test_tavily_http_failure_is_reported_without_key(tmp_path, monkeypatch):
    settings = load_settings()
    settings["tavily"]["cache_dir"] = str(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")

    def failed(*_args, **_kwargs):
        raise HTTPError("https://api.tavily.com/search", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(core, "_post_json", failed)
    with pytest.raises(ValueError, match="Tavily API returned HTTP 401") as exc:
        core.tavily_research("org/x license", settings)
    assert "test-secret" not in str(exc.value)


def test_final_nemotron_validation_uses_hf_scan_and_downgrades_web_license(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-secret")
    payloads = []
    direct_url = "https://huggingface.co/org/x/tree/abc"

    def fake_post(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return {"content": '{"viable_model_ids":["org/x"]}'}
        return {"content": json.dumps({
            "viable_model_ids": ["org/x"],
            "field_assessments": {
                "org/x": {
                    "license": {
                        "value": "apache-2.0",
                        "status": "verified",
                        "confidence": 0.9,
                        "sources": ["https://example.org/parent"],
                        "action": "accept",
                    }
                }
            },
        })}

    _mock_nebius(monkeypatch, fake_post)
    result = recommend(
        _spec(candidates=["org/x"]),
        load_settings(),
        tmp_path,
        lookup=lambda **_kwargs: [_record("org/x", 3)],
        search=lambda *_args, **_kwargs: [
            {
                "source_url": "https://example.org/parent",
                "origin": "tavily",
                "retrieved_text": "parent license",
            }
        ],
    )
    assert result["recommendation"]["model_id"] == "org/x"
    prevalidation = json.loads(
        (tmp_path / "nemotron_prevalidation_response.json").read_text()
    )
    assert len(prevalidation["completions"]) == 1
    assert prevalidation["completions"][0]["choices"][0]["message"]["content"] == (
        '{"viable_model_ids":["org/x"]}'
    )
    assert prevalidation["final_message"]["content"] == '{"viable_model_ids":["org/x"]}'
    assert payloads[1]["reasoning_effort"] == "minimal"
    assert payloads[1]["max_tokens"] == 3000
    assert "The FINAL ANSWER must contain only one JSON object" in payloads[1]["messages"][0]["content"]
    assert "Run ID:" in payloads[1]["messages"][1]["content"]
    assert "Cite saved source URLs" in payloads[1]["messages"][1]["content"]
    assert direct_url in payloads[1]["messages"][1]["content"]
    assert "test-secret" not in payloads[1]["messages"][1]["content"]
    claim = result["router_assessment"]["field_assessments"]["org/x"]["license"]
    assert claim["status"] == "inferred"
    assert claim["action"] == "needs_direct_hf_check"


def test_validation_only_reuses_saved_evidence_without_lookup_or_search(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-secret")
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return {"content": '{"viable_model_ids":["org/x"]}',
                "reasoning_content": "HF evidence is sufficient."}

    _mock_nebius(monkeypatch, fake_post)
    result = recommend(
        _spec(candidates=["org/x"]),
        load_settings(),
        tmp_path,
        initial_records=[_record("org/x", 3)],
        initial_evidence={"org/x": [{"source_url": "https://example.org", "origin": "tavily"}]},
        validation_only=True,
        lookup=lambda **_kwargs: pytest.fail("HF lookup repeated"),
        search=lambda *_args, **_kwargs: pytest.fail("Tavily search repeated"),
    )
    assert len(calls) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert result["recommendation"]["model_id"] == "org/x"
    response = json.loads((tmp_path / "nemotron_validation_response.json").read_text())
    assert response["message"]["reasoning_content"] == "HF evidence is sufficient."
    assert any(event["event"] == "validation_only" for event in result["events"])


def test_validation_summarizes_chat_template_and_rejects_verified_missing_value(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-secret")
    payloads = []
    template = "{% for message in messages %}" * 300

    def fake_post(payload):
        payloads.append(payload)
        return {"content": json.dumps({
            "viable_model_ids": [],
            "insufficient_evidence": True,
            "field_assessments": {"org/x": {"base_model": {
                "value": None, "status": "verified", "confidence": 1,
                "sources": ["https://huggingface.co/org/x/tree/abc"], "action": "accept",
            }}},
        })}

    _mock_nebius(monkeypatch, fake_post)
    result = recommend(
        _spec(candidates=["org/x"]), load_settings(), tmp_path,
        initial_records=[_record("org/x", 3) | {"tokenizer_config": {"chat_template": template}}],
        validation_only=True,
    )
    prompt = payloads[0]["messages"][1]["content"]
    assert template not in prompt
    assert "sha256" in prompt
    claim = result["router_assessment"]["field_assessments"]["org/x"]["base_model"]
    assert claim["status"] == "unknown"
    assert claim["action"] == "keep_blocked"


def test_sweep_plan_lists_compatible_frameworks(tmp_path):
    settings = load_settings()
    result = recommend(
        _spec(framework_mode="sweep", candidates=["org/x"]),
        settings,
        tmp_path,
        chat=lambda: {"content": '{"viable_model_ids":["org/x"]}'},
        lookup=lambda **_kwargs: [_record("org/x", 3)],
        search=lambda *_args, **_kwargs: [],
    )
    assert result["benchmark_plan"][0]["frameworks"] == ["transformers"]
    assert result["recommendation"]["task_fit"] == "inferred"


def test_alternatives_are_from_distinct_families(tmp_path):
    settings = load_settings()
    records = [_record("org/Qwen3-3B", 3), _record("org/Qwen3-4B", 4), _record("org/Gemma-5B", 5)]
    result = recommend(
        _spec(candidates=["org/Qwen3-3B"]),
        settings,
        tmp_path,
        chat=lambda: {
            "content": json.dumps({"viable_model_ids": [row["model_id"] for row in records]})
        },
        lookup=lambda **_kwargs: records,
        search=lambda *_args, **_kwargs: [],
    )
    assert result["recommendation"]["model_id"] == "org/Qwen3-3B"
    assert [item["model_id"] for item in result["alternatives"]] == ["org/Gemma-5B"]


def test_speculative_decoding_requires_pair_and_supported_runtime():
    assessment = core.runtime_assessment(
        _record("org/x", 3),
        [{"source_url": "https://example.org/spec", "retrieved_text": "EAGLE draft support"}],
    )
    assert assessment["speculative_decoding"]["usable"] is False
    assert assessment["vllm"]["status"] == "unknown"
    record = _record("org/x", 3)
    record["config"]["speculative_decoding"] = {
        "target_model": "org/x",
        "draft_model": "org/draft",
        "runtime": "transformers",
    }
    supported = core.runtime_assessment(record, [])
    assert supported["speculative_decoding"]["usable"] is True


def test_exact_dflash_pair_passes_a100_capacity_with_unmeasured_speedup():
    record = _record("Qwen/Qwen3.5-4B", 4.66) | {
        "estimated_download_size_gb": 8.68,
        "context_length": 2048,
        "context_length_source": "fallback",
    }
    record["config"]["text_config"] = {"max_position_embeddings": 262144}
    source = {
        "source_url": "https://huggingface.co/z-lab/Qwen3.5-4B-DFlash",
        "retrieved_text": "Intro " * 180 + "This repository contains a DFlash draft model for `Qwen/Qwen3.5-4B`. "
        "It is intended to be paired with the target model in an SGLang speculative decoding server.",
    }
    pair = core.runtime_assessment(record, [source])["speculative_decoding"]
    assert pair["usable"] is True
    assert pair["draft_model"] == "z-lab/Qwen3.5-4B-DFlash"
    assert pair["a100_performance"] == "unmeasured"
    assert "draft model for `Qwen/Qwen3.5-4B`" in core._validation_excerpt(source)
    ranked = rank_candidates(
        [record], _spec(request="Find speculative decoding pair", license="any", gpu_hourly_cost_usd=5),
        {record["model_id"]: [source]}, load_settings(),
    )[0]
    assert ranked["eligible"] is True
    assert ranked["runtime"] == "sglang"
    assert ranked["estimated_bf16_memory_gb"] < 80 * 0.85
    assert ranked["context_length"] == 262144


def test_documented_pair_overrides_missing_built_in_head_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-secret")
    _mock_nebius(monkeypatch, lambda _payload: {
        "content": json.dumps({
            "viable_model_ids": [], "insufficient_evidence": True,
            "reasoning": "No built-in speculative head.",
        }),
    })
    record = _record("Qwen/Qwen3.5-4B", 4.66) | {"estimated_download_size_gb": 8.68}
    record["config"]["text_config"] = {"max_position_embeddings": 262144}
    source = {
        "source_url": "https://huggingface.co/z-lab/Qwen3.5-4B-DFlash",
        "retrieved_text": "DFlash draft model for `Qwen/Qwen3.5-4B`. SGLang speculative decoding server.",
    }
    result = recommend(
        _spec(request="Find speculative decoding pair", license="any", gpu_hourly_cost_usd=5),
        load_settings(), tmp_path, initial_records=[record],
        initial_evidence={record["model_id"]: [source]}, validation_only=True,
    )
    assert result["status"] == "recommended"
    assert result["recommendation"]["speculative_decoding"]["draft_model"] == "z-lab/Qwen3.5-4B-DFlash"
    assert source["source_url"] in result["router_assessment"]["reasoning"]


def test_pair_selection_rejects_gguf_copy_and_uses_score_for_native_ties():
    record = _record("Qwen/Qwen3.5-4B", 4.66)
    copied_card = "Quantized version of z-lab/Qwen3.5-4B-DFlash. Their model card is pasted as is below. "
    pairing = "This repository contains a DFlash draft model for `Qwen/Qwen3.5-4B`. "
    evidence = [
        {"source_url": "https://huggingface.co/EntityDeletr/Qwen3.5-4B-DFlash-GGUF",
         "score": 0.99, "retrieved_text": copied_card + pairing + "SGLang DFlash support."},
        {"source_url": "https://huggingface.co/modal-labs/Qwen3.5-4B-DFlash",
         "score": 0.5, "retrieved_text": pairing + "SGLang DFlash support."},
        {"source_url": "https://huggingface.co/z-lab/Qwen3.5-4B-DFlash",
         "score": 0.8, "retrieved_text": pairing + "SGLang DFlash support."},
    ]
    pair = core.runtime_assessment(record, evidence)["speculative_decoding"]
    assert pair["draft_model"] == "z-lab/Qwen3.5-4B-DFlash"
    assert pair["source_url"] == "https://huggingface.co/z-lab/Qwen3.5-4B-DFlash"


def test_benchmark_blocks_unsupported_framework_even_when_confirmed(tmp_path):
    path = tmp_path / "recommendation.json"
    item = {
        "model_id": "org/x",
        "revision": "abc",
        "license": "apache-2.0",
        "blocking_reasons": [],
        "compatible_frameworks": ["sglang"],
        "estimated_bf16_memory_gb": 8,
    }
    path.write_text(
        json.dumps(
            {
                "status": "recommended",
                "recommendation": item,
                "framework_mode": "fast",
                "evaluated_candidates": [item],
                "benchmark_plan": [
                    {
                        "model_id": "org/x",
                        "revision": "abc",
                        "task_fixture": "social_scoring",
                        "frameworks": ["sglang"],
                    }
                ],
            }
        )
    )
    result = benchmark_recommendation(path, confirm=True, submit=lambda *_args, **_kwargs: 1)
    assert result["status"] == "preflight_failed"
    assert result["submitted_jobs"] == []


def test_fixture_benchmark_records_metrics_and_sqlite(tmp_path):
    class Session:
        def __init__(self, *_args):
            pass

        def generate(self, prompt, _limit):
            return GenerationResult(prompt, '{"urgency":"high"}', 40, 8, 0.25)

    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    payload = WorkerJobPayload(
        run_id="benchmark1",
        model_id="org/x",
        revision="abc",
        config={
            "benchmark_recommendation": {
                "framework": "transformers",
                "task_fixture": "social_scoring",
            }
        },
    )
    summary = run_fixture_benchmark(
        payload,
        tmp_path / "benchmark1",
        store,
        session_factory=Session,
        memory_reset=lambda: None,
        memory_read=lambda: None,
    )
    assert summary["fixture_score"] == 1.0
    assert summary["throughput_tokens_per_second_mean"] == 32
    assert (
        json.loads((tmp_path / "benchmark1" / "benchmark_results.json").read_text())["summary"][
            "framework"
        ]
        == "transformers"
    )


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trt-llm"])
def test_framework_server_commands_and_fixture_artifact(framework, tmp_path):
    command = build_server_command(framework, "/pinned/model", 8765, gpu_count=2)
    assert "/pinned/model" in command
    assert "8765" in command
    assert "2" in command

    class Session:
        def __init__(self, *_args):
            pass

        def generate(self, prompt, _limit):
            return GenerationResult(prompt, "high", 30, 1, 0.5)

    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    payload = WorkerJobPayload(
        run_id="framework1",
        model_id="org/x",
        revision="abc",
        config={
            "benchmark_recommendation": {"framework": framework, "task_fixture": "social_scoring"}
        },
    )
    result = run_fixture_benchmark(
        payload,
        tmp_path / "framework1",
        store,
        session_factory=Session,
        memory_reset=lambda: None,
        memory_read=lambda: 4.0,
    )
    assert result["framework"] == framework
    assert result["gpu_peak_memory_gb"] == 4.0


def test_benchmark_report_compares_synced_framework_results(tmp_path):
    artifact = tmp_path / "recommendation.json"
    artifact.write_text("{}")
    jobs = [
        {"run_id": "run_a", "model_id": "org/x", "framework": "transformers"},
        {"run_id": "run_b", "model_id": "org/x", "framework": "vllm"},
    ]
    (tmp_path / "benchmark_submission.json").write_text(json.dumps({"submitted_jobs": jobs}))
    results = tmp_path / "synced"
    for job, latency in zip(jobs, [2.0, 1.0], strict=True):
        folder = results / job["run_id"]
        folder.mkdir(parents=True)
        (folder / "benchmark_results.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "run_id": job["run_id"],
                        "model_id": "org/x",
                        "revision": "abc",
                        "framework": job["framework"],
                        "latency_seconds_mean": latency,
                        "fixture_score": 1.0,
                    }
                }
            )
        )
    report = build_benchmark_report(artifact, results, tmp_path / "report.sqlite")
    assert report["status"] == "complete"
    assert report["comparisons"][0]["fastest_framework"] == "vllm"
    assert (tmp_path / "framework_comparison.json").exists()


@pytest.mark.parametrize(
    ("user_request", "fixture_id", "required_context"),
    [
        ("Review a full pull request", "pr_review", 25500),
        ("Score short social posts", "social_scoring", 320),
        ("Analyze a long document collection", "long_context_analysis", 65200),
    ],
)
def test_fixture_selection_and_context_gate(user_request, fixture_id, required_context):
    fixture = core.select_fixture(user_request)
    assert fixture["id"] == fixture_id
    assert fixture["input_tokens"] + fixture["output_tokens"] == required_context
    candidate = _record("org/x", 3) | {"context_length": required_context - 1}
    ranked = rank_candidates([candidate], _spec(request=user_request), {}, load_settings())
    assert "context_window_unverified_or_insufficient" in ranked[0]["blocking_reasons"]
