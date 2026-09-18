"""Run the recommender's HF, Tavily, and Nemotron stages independently."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from hf_model_watch.recommender.core import (
    RequestSpec,
    hf_lookup,
    load_settings,
    model_research_queries,
    recommend,
    tavily_research,
)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("status", "hf", "tavily", "nemotron"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request", default="Score short social posts with low latency")
    parser.add_argument("--gpu-memory-gb", type=float, default=80)
    parser.add_argument("--budget-usd-per-hour", type=float, default=5)
    parser.add_argument("--gpu-hourly-cost-usd", type=float, default=None)
    parser.add_argument("--license", default="any")
    parser.add_argument("--framework-mode", choices=("fast", "sweep"), default="fast")
    parser.add_argument("--nemotron-model")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        print(
            json.dumps(
                {
                    "nebius_key_present": bool(os.getenv("NEBIUS_API_KEY")),
                    "tavily_key_present": bool(os.getenv("TAVILY_API_KEY")),
                    "output_dir": str(destination),
                }
            )
        )
        return
    settings = load_settings()
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            parser.error("--max-tokens must be greater than zero")
        settings["nebius"]["validation_max_tokens"] = args.max_tokens
    hf_path = destination / "hf_record.json"
    tavily_path = destination / "tavily_evidence.json"
    if args.stage == "hf":
        records = hf_lookup(model_id=args.model_id, artifact_dir=destination, limit=1)
        record = records[0]
        _write(hf_path, record)
        print(
            json.dumps(
                {
                    "model_id": record["model_id"],
                    "revision": record.get("revision"),
                    "license": record.get("license"),
                    "parameters_billion": record.get("estimated_params_billion"),
                    "context_length": record.get("context_length"),
                    "architecture": record.get("architecture"),
                    "chat_template_present": bool(
                        (record.get("tokenizer_config") or {}).get("chat_template")
                    ),
                    "download_size_gb": record.get("estimated_download_size_gb"),
                    "missing_fields": record.get("missing_fields"),
                    "card_artifact": record.get("card_artifact"),
                    "record_artifact": str(hf_path),
                },
                default=str,
            )
        )
        return
    if not hf_path.exists():
        parser.error(f"Run the hf stage first; missing {hf_path}")
    record = json.loads(hf_path.read_text(encoding="utf-8"))
    if args.stage == "tavily":
        if not os.getenv("TAVILY_API_KEY"):
            _write(tavily_path, [])
            _write(destination / "tavily_searches.json", [])
            print(
                json.dumps(
                    {
                        "status": "missing_api_key",
                        "message": "Add TAVILY_API_KEY to the repository .env and rerun.",
                        "evidence_artifact": str(tavily_path),
                    }
                )
            )
            return
        evidence = []
        cases = []
        for name, query in model_research_queries(record, args.request)[
            : int(settings["tavily"]["max_searches"])
        ]:
            try:
                found = tavily_research(query, settings)
                for source in found:
                    source["research_case"] = name
                evidence.extend(found)
                cases.append({"case": name, "query": query, "count": len(found)})
            except ValueError as exc:
                cases.append({"case": name, "query": query, "error": str(exc)[:200]})
        _write(tavily_path, evidence)
        _write(destination / "tavily_searches.json", cases)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "cases": cases,
                    "source_count": len(evidence),
                    "top_sources": [
                        {"url": row["source_url"], "claim": row["claim"]} for row in evidence[:5]
                    ],
                    "evidence_artifact": str(tavily_path),
                }
            )
        )
        return
    if not os.getenv("NEBIUS_API_KEY"):
        parser.error("NEBIUS_API_KEY is missing from the WSL process and repository .env")
    evidence = json.loads(tavily_path.read_text(encoding="utf-8")) if tavily_path.exists() else []
    spec = RequestSpec(
        request=args.request,
        gpu_memory_gb=args.gpu_memory_gb,
        budget_usd_per_hour=args.budget_usd_per_hour,
        gpu_hourly_cost_usd=args.gpu_hourly_cost_usd,
        license=args.license,
        framework_mode=args.framework_mode,
        model=args.nemotron_model,
        candidates=[record["model_id"]],
    )
    result = recommend(
        spec,
        settings,
        destination,
        initial_records=[record],
        initial_evidence={record["model_id"]: evidence},
        validation_only=True,
    )
    response_path = destination / "nemotron_validation_response.json"
    response = (
        json.loads(response_path.read_text(encoding="utf-8")) if response_path.exists() else {}
    )
    message = response.get("message", response)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": (result["recommendation"] or {}).get("model_id"),
                "router_assessment": result["router_assessment"],
                "candidate_blockers": result["evaluated_candidates"][0]["blocking_reasons"],
                "events": result["events"],
                "reasoning_excerpt": str(
                    message.get("reasoning") or message.get("reasoning_content") or ""
                )[:800],
                "response_excerpt": str(message.get("content") or "")[:800],
                "finish_reason": response.get("finish_reason"),
                "response_artifact": str(response_path) if response_path.exists() else None,
                "artifact_path": str(destination / "recommendation.json"),
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
