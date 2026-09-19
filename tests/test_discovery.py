from __future__ import annotations

import json

from tests.test_recommender import _spec
from typer.testing import CliRunner

from hf_model_watch.cli import app, format_fit_checks
from hf_model_watch.recommender import discovery
from hf_model_watch.recommender.core import load_settings


def test_explicit_and_discovery_render_shared_python_fit_checks():
    assessment = {
        "eligible": True,
        "blocking_reasons": [],
        "estimated_bf16_memory_gb": 24,
        "runtime": "vllm",
        "compatible_frameworks": ["vllm", "transformers"],
        "context_length": 32768,
        "license": "apache-2.0",
        "estimated_hourly_usd": 3,
    }
    explicit_lines = format_fit_checks(assessment, _spec())
    discovery_recommendation = {"runtime_assessment": assessment}
    discovery_lines = format_fit_checks(discovery_recommendation["runtime_assessment"], _spec())

    assert explicit_lines == discovery_lines
    rendered = "\n".join(explicit_lines)
    for field in (
        "Eligibility: eligible",
        "Blockers: none",
        "Memory: 24 GB estimated / 80 GB per GPU (1 GPU) available",
        "Runtime: vllm",
        "Compatible frameworks: vllm, transformers",
        "Context length: 32768",
        "License result: passes (apache-2.0)",
        "Budget result: passes ($3/hour against $5/hour)",
    ):
        assert field in rendered


def test_explore_family_records_filters_scores_and_deduplicates(tmp_path, monkeypatch):
    requested = {
        "model_id": "org/Qwen3-4B",
        "estimated_params_billion": 4,
        "created_at": "2025-04-28T10:00:00+00:00",
    }
    rows = [
        requested,
        {"model_id": "org/Qwen3.5-4B", "estimated_params_billion": 4,
         "created_at": "2025-05-01T10:00:00Z"},
        {"model_id": "org/Qwen2.5-refresh-5B", "estimated_params_billion": 5,
         "created_at": "2025-05-02T10:00:00+00:00"},
        {"model_id": "org/Qwen4-4B", "estimated_params_billion": 4,
         "created_at": "2025-04-01T10:00:00+00:00"},
        {"model_id": "org/Qwen4-20B", "estimated_params_billion": 20,
         "created_at": "2025-05-03T10:00:00+00:00"},
        {"model_id": "org/Qwen-Drive-1.0-4B", "estimated_params_billion": 4,
         "created_at": "2025-05-04T10:00:00+00:00", "downloads": 1000000},
    ]
    monkeypatch.setattr(
        discovery, "score_candidate",
        lambda row, *_args: {"eligible": "20B" not in row["model_id"], "score": 1 if "3.5" in row["model_id"] else .5},
    )
    progress = []
    lookup_calls = []
    def lookup(**kwargs):
        lookup_calls.append(kwargs)
        if kwargs.get("model_id"):
            return [next(row for row in rows if row["model_id"] == kwargs["model_id"])]
        return rows
    result = discovery.explore_family_records(
        requested, _spec(), load_settings(), tmp_path,
        lookup=lookup, progress=progress.append,
    )
    assert [row["model_id"] for row in result] == ["org/Qwen3.5-4B", "org/Qwen2.5-refresh-5B"]
    assert any("Family lookup" in item for item in progress)
    assert any("deduplication" in item for item in progress)
    assert lookup_calls[0]["author"] == "org"
    assert lookup_calls[0]["limit"] == 100
    assert lookup_calls[0]["sort"] == "createdAt"
    assert lookup_calls[0]["enrich"] is False
    assert len([call for call in lookup_calls if call.get("model_id")]) == 2


def test_constraints_and_missing_inputs_reuse_request():
    assert discovery.extract_constraints(
        "Want to run fast Qwen3.5-4b for log analysis"
    )["family"] == "Qwen"
    request = (
        "Find a new model like Qwen3.5-4B with speculative decoding for one 80GB A100, "
        "budget $5/hour; any license, sweep mode"
    )
    parsed = discovery.extract_constraints(request)
    assert parsed["reference_model"] == "Qwen3.5-4B"
    assert parsed["family"] == "Qwen"
    assert parsed["target_params_billion"] == 4
    assert "coding" not in parsed["features"]
    assert (
        "speculative decoding"
        in discovery.extract_constraints("Find a model with Medusa head")["features"]
    )
    assert (
        "custom routing"
        in discovery.extract_constraints("Find a model with custom routing head")["features"]
    )
    assert (
        discovery.extract_constraints("maximum of 2000 generated tokens")["max_generation_tokens"]
        == 2000
    )
    values, missing = discovery.discovery_inputs(request, {"gpu_hourly_cost_usd": 3})
    assert values["gpu_memory_gb"] == 80
    assert values["budget_usd_per_hour"] == 5
    assert values["gpu_hourly_cost_usd"] == 3
    assert not missing


def test_discovery_merges_scans_and_researches_original_request(tmp_path, monkeypatch):
    rows = {
        name: {"model_id": name, "estimated_params_billion": 4, "estimated_download_size_gb": 8}
        for name in ("Qwen/a", "Qwen/b", "google/gemma")
    }
    calls = []
    progress_events = []

    def scan(*_args):
        return [rows["Qwen/a"], rows["Qwen/b"]]

    def lookup(*, family, **_kwargs):
        return [rows["Qwen/a"]] if family == "Qwen" else [rows["google/gemma"]]

    def search(query, _settings, *, extract_urls=None):
        calls.append((query, extract_urls))
        return [
            {
                "source_url": "https://huggingface.co/Qwen/a",
                "retrieved_text": "Qwen/a speculative decoding",
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(
        discovery,
        "score_candidate",
        lambda row, *_args: {"model_id": row["model_id"], "score": 0.7, "eligible": True},
    )
    monkeypatch.setattr(
        discovery,
        "rank_candidates",
        lambda records, *_args: [
            {
                "model_id": row["model_id"],
                "eligible": True,
                "blocking_reasons": [],
                "estimated_bf16_memory_gb": 24,
                "compatible_frameworks": ["transformers"],
            }
            for row in records
        ],
    )
    spec = _spec(request="Find a new model like Qwen3.5-4B with speculative decoding")
    result = discovery.discover_models(
        spec,
        load_settings(),
        tmp_path,
        scan=scan,
        lookup=lookup,
        search=search,
        progress=progress_events.append,
    )
    assert len(result["recommendations"]) == 3
    assert len({row["model_id"] for row in result["recommendations"]}) == 3
    assert set(
        next(row for row in result["recommendations"] if row["model_id"] == "Qwen/a")[
            "discovery_strategies"
        ]
    ) == {"recent_similar", "same_family"}
    assert all(spec.request in query for query, urls in calls if "User requirement:" in query)
    assert any(urls for _, urls in calls)
    assert any("Tavily Search" in event for event in progress_events)
    assert any("Scoring" in event for event in progress_events)
    assert result["scan_diagnostics"]["gemma"]["found"] == 1
    gemma = next(row for row in result["recommendations"] if row["model_id"] == "google/gemma")
    assert gemma["evidence"] == []  # Qwen/a pages cannot support Gemma claims.


def test_cli_selection_and_explicit_branch(tmp_path, monkeypatch):
    import hf_model_watch.recommender.core as core

    called = []

    def explicit(spec, _settings, _destination, **_kwargs):
        called.append(spec.candidates)
        return {"status": "recommended", "recommendation": {"model_id": "org/exact"}}

    monkeypatch.setattr(core, "recommend", explicit)
    args = [
        "recommend",
        "Find a model",
        "--model-id",
        "org/exact",
        "--gpu-memory-gb",
        "80",
        "--budget-usd-per-hour",
        "5",
        "--gpu-hourly-cost-usd",
        "3",
        "--license",
        "any",
        "--framework-mode",
        "sweep",
        "--output-dir",
        str(tmp_path),
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    assert called == [["org/exact"]]
    monkeypatch.setattr(
        discovery,
        "discover_models",
        lambda *_args, **_kwargs: {
            "status": "recommended",
            "recommendations": [
                {"model_id": "org/a", "hf_score": 0.8, "why_it_fits": ["fits"], "limitations": []}
            ],
            "selected_model_id": None,
        },
    )
    result = CliRunner().invoke(
        app, [arg for arg in args if arg not in ("--model-id", "org/exact")], input="1\n"
    )
    assert result.exit_code == 0
    assert "Selected model: org/a" in result.output
    assert "Optimize Inference + Benchmark" in result.output
    assert json.loads((tmp_path / "discovery.json").read_text())["selected_model_id"] == "org/a"


def test_discovery_json_stdout_excludes_scanner_trace(tmp_path, monkeypatch):
    def fake_discovery(_spec, _settings, destination, **_kwargs):
        destination.mkdir(parents=True, exist_ok=True)
        print('TRACE_EVENT {"event_type":"scan_started"}')
        return {"status": "insufficient_evidence", "recommendations": []}

    monkeypatch.setattr(discovery, "discover_models", fake_discovery)
    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "Find a model",
            "--gpu-memory-gb",
            "80",
            "--budget-usd-per-hour",
            "5",
            "--gpu-hourly-cost-usd",
            "3",
            "--license",
            "any",
            "--framework-mode",
            "sweep",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "insufficient_evidence"
    assert "TRACE_EVENT" not in result.stdout


def test_candidate_evidence_excludes_other_model_pages():
    assert not discovery._candidate_source(
        {
            "source_url": "https://huggingface.co/Qwen/Qwen3.5-4B",
            "retrieved_text": "Qwen3-4B is mentioned in a comparison",
        },
        "Qwen/Qwen3-4B",
    )
    assert discovery._candidate_source(
        {"source_url": "https://huggingface.co/Qwen/Qwen3-4B"},
        "Qwen/Qwen3-4B",
    )


def test_discovery_researches_exact_dflash_draft_pair(tmp_path, monkeypatch):
    row = {
        "model_id": "Qwen/Qwen3-4B",
        "revision": "abc",
        "license": "apache-2.0",
        "estimated_params_billion": 4,
        "estimated_download_size_gb": 8,
        "has_config": True,
        "has_weights": True,
        "config": {"architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 131072},
        "context_length": 131072,
        "context_length_source": "metadata",
    }
    draft_url = "https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16"
    draft_page = {
        "source_url": draft_url,
        "retrieved_text": (
            "This DFlash drafter must be used with target model `Qwen/Qwen3-4B`. "
            "Run with SGLang and --speculative-draft-model-path z-lab/Qwen3-4B-DFlash-b16."
        ),
    }
    queries = []
    extracted_urls = []

    def search(query, _settings, *, extract_urls=None):
        queries.append(query)
        if extract_urls:
            extracted_urls.extend(extract_urls)
            return [draft_page] if draft_url in extract_urls else []
        return (
            [
                {
                    "source_url": "https://huggingface.co/z-lab/Qwen3.5-4B-DFlash",
                    "retrieved_text": "other target",
                },
                {
                    "source_url": "https://huggingface.co/other/Qwen3-4B-DFlash-GGUF",
                    "retrieved_text": "converted copy",
                },
                {"source_url": draft_url, "retrieved_text": "DFlash draft model"},
            ]
            if "speculative decoding draft target model" in query
            else []
        )

    monkeypatch.setattr(
        discovery,
        "score_candidate",
        lambda candidate, *_args: {
            "model_id": candidate["model_id"],
            "score": 0.8,
            "eligible": True,
        },
    )
    spec = _spec(
        request="Find a model like Qwen3.5-4B with optional speculative decoding head",
        license="any",
    )
    result = discovery.discover_models(
        spec,
        load_settings(),
        tmp_path,
        scan=lambda *_args: [row],
        lookup=lambda **_kwargs: [],
        search=search,
    )
    recommended = result["recommendations"][0]
    focused_query = next(query for query in queries if "draft target model" in query)
    assert "Qwen/Qwen3-4B" in focused_query
    assert "Qwen3.5-4B" not in focused_query
    assert "DFlash" not in focused_query
    assert any(spec.request in query for query in queries if "User requirement:" in query)
    assert recommended["task_fit"]["speculative_decoding"] == "verified"
    assert "Qwen/Qwen3-4B" in recommended["reasoning"]
    assert "score of 0.800" in recommended["reasoning"]
    assert "z-lab/Qwen3-4B-DFlash-b16" in recommended["reasoning"]
    assert recommended["runtime_assessment"]["speculative_decoding"]["draft_model"] == (
        "z-lab/Qwen3-4B-DFlash-b16"
    )
    assert draft_url in [source["source_url"] for source in recommended["evidence"]]
    assert draft_url in extracted_urls
    assert not any("GGUF" in url for url in extracted_urls)


def test_generic_draft_pair_and_feature_evidence():
    from hf_model_watch.recommender.core import runtime_assessment

    record = {
        "model_id": "org/target-4B",
        "has_config": True,
        "has_weights": True,
        "config": {"architectures": ["ExampleForCausalLM"]},
    }
    draft = {
        "source_url": "https://huggingface.co/org/target-4B-DSpark",
        "retrieved_text": (
            "DSpark speculative draft model for `org/target-4B`. "
            "Use SGLang with --speculative-draft-model-path org/target-4B-DSpark."
        ),
    }
    assert runtime_assessment(record, [draft])["speculative_decoding"]["draft_model"] == (
        "org/target-4B-DSpark"
    )
    target_page = {
        "source_url": "https://huggingface.co/org/target-4B",
        "retrieved_text": "org/target-4B supports tool calling with a chat template.",
    }
    assert discovery._feature_status("tool calling", "org/target-4B", [target_page]) == "verified"
    assert discovery._feature_status("vision", "org/target-4B", [target_page]) == "unknown"


def test_discovery_researches_requested_non_speculative_feature(tmp_path, monkeypatch):
    row = {
        "model_id": "org/vision-4B",
        "revision": "abc",
        "license": "apache-2.0",
        "estimated_params_billion": 4,
        "estimated_download_size_gb": 8,
        "has_config": True,
        "has_weights": True,
        "config": {"architectures": ["ExampleForCausalLM"], "max_position_embeddings": 131072},
        "context_length": 131072,
        "context_length_source": "metadata",
    }
    page = {
        "source_url": "https://huggingface.co/org/vision-4B",
        "retrieved_text": "org/vision-4B supports vision and image inputs.",
    }
    queries = []

    def search(query, _settings, *, extract_urls=None):
        queries.append(query)
        return [page] if extract_urls or "vision support documentation" in query else []

    monkeypatch.setattr(
        discovery,
        "score_candidate",
        lambda candidate, *_args: {
            "model_id": candidate["model_id"],
            "score": 0.8,
            "eligible": True,
        },
    )
    result = discovery.discover_models(
        _spec(request="Find a vision model like Vision4B", license="any"),
        load_settings(),
        tmp_path,
        scan=lambda *_args: [row],
        lookup=lambda **_kwargs: [],
        search=search,
    )
    assert any("vision support documentation" in query for query in queries)
    assert all("DFlash" not in query for query in queries)
    assert result["recommendations"][0]["task_fit"]["vision"] == "verified"


def test_nemotron_discovery_claim_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    settings = load_settings()
    base = {
        "model_id": "org/a", "eligible": True, "hf_score": 0.8,
        "final_score": 0.9, "runtime_assessment": {"blocking_reasons": []},
        "task_fit": {"tool_calling": "verified"},
        "evidence": [{"source_url": "https://huggingface.co/org/a",
                      "retrieved_text": "org/a supports tool calling"}],
    }
    response = {"assessments": [{
        "model_id": "org/a", "task_fit": "verified", "rationale": "Tool calling is documented.",
        "uncertainty": "Speed unmeasured", "claims": [{
            "feature": "tool_calling", "status": "verified",
            "source_urls": ["https://huggingface.co/org/a"],
        }],
    }]}
    calls = []

    def fake_client(**kwargs):
        def create(**request):
            calls.append(request)
            message = SimpleNamespace(model_dump=lambda **_: {"content": json.dumps(response)})
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(discovery, "OpenAI", fake_client)
    row = dict(base)
    discovery._validate_with_nemotron(_spec(), [row], settings, tmp_path)
    assert row["nemotron"]["task_fit"] == "verified"
    assert row["final_score"] == 0.92
    assert calls[0]["model"] == settings["nebius"]["default_model"]
    assert "request" in calls[0]["messages"][1]["content"]
    assert "hf_score" in calls[0]["messages"][1]["content"]
    assert "deployment" in calls[0]["messages"][1]["content"]
    assert "https://huggingface.co/org/a" in calls[0]["messages"][1]["content"]
    assert (tmp_path / "nemotron_discovery_response.json").exists()

    response["assessments"][0]["claims"][0]["source_urls"] = ["https://elsewhere.invalid"]
    row = dict(base)
    discovery._validate_with_nemotron(_spec(), [row], settings, tmp_path)
    assert row["nemotron"]["task_fit"] == "unknown"
    assert row["nemotron"]["unsupported_claims"]
    assert row["final_score"] == 0.9

    response["assessments"][0]["claims"][0]["source_urls"] = []
    row = {**base, "evidence": []}
    discovery._validate_with_nemotron(_spec(), [row], settings, tmp_path)
    assert row["nemotron"]["task_fit"] == "unknown"
    assert row["nemotron"]["rationale"] == ""
