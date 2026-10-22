"""Explicitly curate optimizer JSONL into sanitized public inference snapshots.

Never scans directories, launches jobs, or publishes automatically. Requires an
operator-selected source list and complete measurement metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def normalize(row: dict, source_sha256: str, index: int) -> dict:
    metadata = row.get("metadata", {})
    results = row.get("results", {})
    config = row.get("config", {})
    client = config.get("client_args", {})
    revision = str(metadata.get("model_revision", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("An immutable model revision is required")
    if metadata.get("measurement") != "MEASURED" or results.get("benchmark_valid") is not True:
        raise ValueError("A valid measured benchmark is required")
    gpu = metadata.get("gpu_type")
    version = metadata.get("framework_version")
    if any(not isinstance(value, str) or not value or re.search(r"placeholder|replace|unknown|^none$|^null$", value, re.I) for value in [gpu, version]):
        raise ValueError("Observed GPU and framework version must be known")
    if any(char in gpu for char in ["/", "\\"]) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", version):
        raise ValueError("Public runtime identity must not contain internal paths")
    model = metadata.get("target_model") or metadata.get("model")
    if not isinstance(model, str) or model.startswith(("/", "\\")) or not re.fullmatch(r"[\w.-]+/[\w.-]+", model):
        raise ValueError("A public model repository identity is required")
    gpu_count = metadata.get("gpu_count")
    if not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("GPU count must be positive")
    workload = {key: client.get(key) for key in ["random_input_len", "random_output_len", "num_prompts", "max_concurrency", "seed"]}
    if any(not isinstance(value, int) or value < (0 if key == "seed" else 1) for key, value in workload.items()):
        raise ValueError("Complete fixed workload metadata is required")
    request_hash = metadata.get("requests_sha256") or metadata.get("workload_sha256")
    if not isinstance(request_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", request_hash):
        raise ValueError("A workload fingerprint is required")
    completed = results.get("completed")
    if completed != workload["num_prompts"] or results.get("failed", 0) != 0:
        raise ValueError("Partial or failed benchmarks are not publishable")
    metrics = {"latency": results.get("mean_e2e_latency_ms"), "throughput": results.get("output_throughput"), "ttft": results.get("mean_ttft_ms")}
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in metrics.values()):
        raise ValueError("Positive finite inference metrics are required")
    engine = config.get("framework")
    if engine not in {"vllm", "sglang", "trtllm"}:
        raise ValueError("Unsupported engine identity")
    context = {"model": model, "revision": revision, "gpu": gpu, "gpuCount": gpu_count, "workload": workload, "requestsSha256": request_hash}
    cohort = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]
    allowed = {"max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization", "max_running_requests", "chunked_prefill_size", "mem_fraction_static", "max_batch_size", "max_num_tokens", "dtype", "kv_cache_dtype", "tensor_parallel_size"}
    settings = {k: v for k, v in config.get("server_args", {}).items() if k in allowed and isinstance(v, (int, float, bool, str)) and (not isinstance(v, str) or re.fullmatch(r"[\w.-]+", v))}
    return {"id": f"{cohort}-{index:04}", "kind": "measurement", "cohort": cohort, "context": context, "engine": {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM"}[engine], "engineVersion": version, "configuration": settings, "sourceSha256": source_sha256, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.input:
        raw = path.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            if line.strip():
                records.append(normalize(json.loads(line), fingerprint, len(records) + 1))
    if not records:
        raise ValueError("No inference measurements selected")
    # Preserve the repository's Pareto semantics for real measurements.
    from hf_model_watch.optimization.analysis import pareto_front
    from hf_model_watch.optimization.contracts import Candidate
    for cohort in {r["cohort"] for r in records}:
        group = [r for r in records if r["cohort"] == cohort]
        candidates = [Candidate(configuration_id=r["id"], framework=r["engine"], configuration=r["configuration"], metrics={"mean_e2e_latency_ms": r["latency"], "output_throughput": r["throughput"]}) for r in group]
        ids = {c.configuration_id for c in pareto_front(candidates, ["output_throughput", "mean_e2e_latency_ms"])}
        for record in group:
            record["pareto"] = record["id"] in ids
    output = {"schemaVersion": 1, "kind": "measurement", "rows": records, "note": "Curated snapshot; review before publication. Not independently validated deployment recommendations."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
