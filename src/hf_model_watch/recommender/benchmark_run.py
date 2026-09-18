from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from hf_model_watch.recommender.core import select_fixture
from hf_model_watch.recommender.server_runtime import OpenAIServerSession, read_server_gpu_memory
from hf_model_watch.runtime.backend_transformers import TransformersSession
from hf_model_watch.storage.schemas import (
    RunManifest,
    initialize_run_artifacts,
    update_manifest_artifact,
    utc_now_iso,
)


def run_fixture_benchmark(
    payload: Any,
    run_dir: Path,
    store: Any,
    session_factory: Any = None,
    memory_reset: Any = None,
    memory_read: Any = None,
) -> dict[str, Any]:
    details = payload.config.get("benchmark_recommendation") or {}
    framework = details.get("framework")
    if framework not in {"transformers", "vllm", "sglang", "trt-llm"}:
        raise ValueError(f"No benchmark execution backend for {framework}")
    memory_reset = memory_reset or (
        _reset_gpu_memory if framework == "transformers" else lambda: None
    )
    memory_read = memory_read or (
        _read_gpu_memory if framework == "transformers" else read_server_gpu_memory
    )
    if not payload.revision:
        raise ValueError("Benchmark requires a pinned revision")
    fixture_id = details.get("task_fixture")
    if fixture_id not in {"pr_review", "social_scoring", "long_context_analysis"}:
        raise ValueError("Unknown benchmark task fixture")
    fixture = select_fixture(
        {
            "pr_review": "pull request",
            "social_scoring": "social post",
            "long_context_analysis": "long context",
        }[fixture_id]
    )
    manifest = RunManifest(
        run_id=payload.run_id,
        model_id=payload.model_id,
        revision=payload.revision,
        config_snapshot={
            "benchmark_recommendation": {"framework": framework, "task_fixture": fixture_id}
        },
    )
    initialize_run_artifacts(run_dir.parent, manifest)
    store.start_run(payload.run_id, payload.model_id, payload.revision, artifact_uri=str(run_dir))
    if session_factory is not None:
        session = session_factory(
            payload.model_id, payload.revision, {"framework": framework, "torch_dtype": "bf16"}
        )
    elif framework == "transformers":
        session = TransformersSession(
            payload.model_id, payload.revision, {"trust_remote_code": False, "torch_dtype": "bf16"}
        )
    else:
        session = OpenAIServerSession(
            payload.model_id, payload.revision, framework, int(details.get("gpu_count") or 1)
        )
    records: list[dict[str, Any]] = []
    for index, example in enumerate(fixture["examples"]):
        try:
            memory_reset()
            generation = session.generate(
                example["prompt"], min(int(fixture["output_tokens"]), 256)
            )
            memory_gb = memory_read()
            text = generation.text
            expected = example["expected_contains"]
            score = sum(word.lower() in text.lower() for word in expected) / len(expected)
            elapsed = float(generation.elapsed_seconds)
            records.append(
                {
                    "example_id": index,
                    "success": True,
                    "generated_text": text,
                    "latency_seconds": elapsed,
                    "throughput_tokens_per_second": generation.generated_tokens / elapsed
                    if elapsed > 0
                    else None,
                    "gpu_peak_memory_gb": memory_gb,
                    "task_score": score,
                    "prompt_tokens": generation.prompt_tokens,
                    "generated_tokens": generation.generated_tokens,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "example_id": index,
                    "success": False,
                    "failure": f"{type(exc).__name__}: {exc}"[:300],
                    "task_score": 0.0,
                }
            )
    if hasattr(session, "close"):
        session.close()
    successes = [item for item in records if item["success"]]
    summary = {
        "run_id": payload.run_id,
        "model_id": payload.model_id,
        "revision": payload.revision,
        "framework": framework,
        "task_fixture": fixture_id,
        "fixture_score": statistics.mean(item["task_score"] for item in records)
        if records
        else 0.0,
        "latency_seconds_mean": statistics.mean(item["latency_seconds"] for item in successes)
        if successes
        else None,
        "throughput_tokens_per_second_mean": statistics.mean(
            item["throughput_tokens_per_second"]
            for item in successes
            if item["throughput_tokens_per_second"] is not None
        )
        if any(item["throughput_tokens_per_second"] is not None for item in successes)
        else None,
        "gpu_peak_memory_gb": max(
            (
                item["gpu_peak_memory_gb"]
                for item in successes
                if item["gpu_peak_memory_gb"] is not None
            ),
            default=None,
        ),
        "failures": len(records) - len(successes),
        "examples": len(records),
    }
    path = run_dir / "benchmark_results.json"
    path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_manifest_artifact(run_dir, manifest, "benchmark_results", str(path))
    (run_dir / "trace_events.jsonl").write_text(
        json.dumps(
            {
                "timestamp": utc_now_iso(),
                "event_type": "fixture_benchmark_completed",
                "stage": "benchmark",
                "status": "failed" if summary["failures"] else "completed",
                "metrics": {
                    "fixture_score": summary["fixture_score"],
                    "failures": summary["failures"],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store.record_runtime_result(
        payload.run_id,
        payload.model_id,
        payload.revision,
        {"backend": framework, "dtype": "bf16", "quantization": None},
        {"fixture": fixture_id, "framework": framework},
    )
    store.record_latency_results(
        payload.run_id, payload.model_id, payload.revision, summary, records
    )
    store.record_quality_results(
        payload.run_id,
        payload.model_id,
        payload.revision,
        {"fixture_score": summary["fixture_score"]},
        records,
    )
    store.record_artifact(
        payload.run_id, "benchmark_results", str(path), payload.model_id, payload.revision
    )
    if summary["failures"]:
        store.record_failure(
            payload.run_id,
            payload.model_id,
            payload.revision,
            "benchmark_generation_failed",
            "One or more fixture examples failed",
        )
    else:
        store.complete_run(
            payload.run_id,
            selected_backend=framework,
            base_score=summary["fixture_score"],
            latency_summary=json.dumps(summary),
        )
    return summary


def _reset_gpu_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _read_gpu_memory() -> float | None:
    import torch

    return torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
