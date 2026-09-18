"""Task-led Hub discovery and evidence-backed model comparison."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hf_model_watch.config import load_yaml_file
from hf_model_watch.hub.lookup import lookup_models
from hf_model_watch.hub.scan import scan_models
from hf_model_watch.hub.scoring import score_candidate
from hf_model_watch.recommender.core import (
    RequestSpec,
    _pair_evidence,
    rank_candidates,
    tavily_research,
)


def extract_constraints(request: str) -> dict[str, Any]:
    text = request.lower()
    features = [
        term
        for term in ("speculative decoding", "reasoning", "vision", "tool calling", "coding")
        if re.search(r"\b" + re.escape(term) + r"\b", text)
    ]
    if (
        re.search(
            r"\b(?:dflash|d-spark|dspark|medusa|eagle(?:-\d+)?|"
            r"multi-token prediction|mtp|draft model)\b",
            text,
        )
        and "speculative decoding" not in features
    ):
        features.append("speculative decoding")
    for match in re.finditer(
        r"\b(?:with|supports?|including)\s+(?:an?\s+)?(?:optional\s+)?"
        r"([\w.+-]+(?:\s+[\w.+-]+){0,3})\s+head\b",
        text,
    ):
        feature = match.group(1).strip()
        if (
            feature not in features
            and "speculative decoding" not in feature
            and feature not in {"dflash", "dspark", "d-spark", "medusa", "eagle", "mtp"}
        ):
            features.append(feature)
    reference = re.search(r"\b(?:like|similar to|based on)\s+([\w.-]+(?:/[\w.-]+)?)", request, re.I)
    reference_model = reference.group(1).rstrip(".,;") if reference else None
    size = re.search(r"\b(\d+(?:\.\d+)?)\s*b\b", reference_model or request, re.I)
    memory = re.search(r"\b(\d+(?:\.\d+)?)\s*gb\b", request, re.I)
    budget = re.search(
        r"(?:budget|maximum inference budget)[^$\n]{0,35}"
        r"\$\s*(\d+(?:\.\d+)?)",
        request,
        re.I,
    )
    cost = re.search(
        r"(?:gpu hourly cost|gpu cost|rental cost)[^$\n]{0,20}\$\s*(\d+(?:\.\d+)?)", request, re.I
    )
    tokens = re.search(
        r"(?:maximum|max|up to)\s+(?:of\s+)?(\d[\d,]*)\s+(?:generated\s+)?tokens", request, re.I
    )
    license_match = re.search(r"\b(any|apache-2\.0|mit|llama|gemma)\s+license\b", request, re.I)
    family = None
    if reference_model:
        family = re.split(r"[-_.]?\d", reference_model.rsplit("/", 1)[-1], maxsplit=1)[0]
    else:
        family_match = re.search(r"\b(qwen|llama|mistral|gemma|phi|deepseek)\b", request, re.I)
        family = family_match.group(1) if family_match else None
    return {
        "reference_model": reference_model,
        "family": family,
        "target_params_billion": float(size.group(1)) if size else None,
        "features": features,
        "use_case": re.search(
            r"(?:for|to)\s+(.+?)(?:\s+(?:with|on|using|for one)\b|[,;.]|$)", request, re.I
        ).group(1)
        if re.search(r"(?:for|to)\s+(.+?)(?:\s+(?:with|on|using|for one)\b|[,;.]|$)", request, re.I)
        else None,
        "gpu_type": (re.search(r"\b(?:A100|H100|L40S|RTX\s*\d{4})\b", request, re.I) or [None])[0],
        "gpu_memory_gb": float(memory.group(1)) if memory else None,
        "budget_usd_per_hour": float(budget.group(1)) if budget else None,
        "gpu_hourly_cost_usd": float(cost.group(1)) if cost else None,
        "max_generation_tokens": int(tokens.group(1).replace(",", "")) if tokens else None,
        "license": license_match.group(1).lower() if license_match else None,
        "framework_mode": "sweep"
        if "sweep mode" in text
        else "fast"
        if "fast mode" in text
        else None,
        "framework": next(
            (name for name in ("vllm", "sglang", "transformers", "trt-llm") if name in text), None
        ),
        "recent": any(term in text for term in ("new", "recent", "latest")),
    }


def discovery_inputs(request: str, supplied: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    parsed = extract_constraints(request)
    values = {"request": request}
    for key in (
        "gpu_memory_gb",
        "budget_usd_per_hour",
        "gpu_hourly_cost_usd",
        "license",
        "framework_mode",
    ):
        values[key] = supplied.get(key) if supplied.get(key) is not None else parsed.get(key)
    # Cost is a user price assumption, never inferred from GPU memory or budget.
    missing = [key for key, value in values.items() if value is None]
    return values, missing


def _approximately_sized(row: dict[str, Any], target: float | None) -> bool:
    if target is None:
        return True
    size = row.get("estimated_params_billion")
    return size is not None and target / 2 <= float(size) <= target * 2


def _candidate_source(source: dict[str, Any], model_id: str) -> bool:
    """Only attach web evidence that names the candidate, not its reference model."""
    url = str(source.get("source_url") or "").lower()
    if "huggingface.co/" in url:
        repository = url.split("huggingface.co/", 1)[1].split("?", 1)[0].split("#", 1)[0]
        repository = "/".join(repository.split("/")[:2])
        if repository != model_id.lower():
            return False
    corpus = " ".join(
        str(source.get(key) or "") for key in ("source_url", "title", "retrieved_text", "claim")
    ).lower()
    stem = model_id.rsplit("/", 1)[-1].lower()
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"(?![a-z0-9])", corpus))


def _hf_repository_url(url: str) -> str | None:
    match = re.match(r"https://huggingface\.co/([^/?#]+/[^/?#]+)", url, re.I)
    return f"https://huggingface.co/{match.group(1)}" if match else None


def _feature_terms(feature: str) -> tuple[str, ...]:
    return {
        "speculative decoding": ("speculat", "draft", "dflash", "dspark", "eagle", "medusa"),
        "tool calling": ("tool calling", "function calling", "tool use"),
        "vision": ("vision", "image", "multimodal"),
        "reasoning": ("reasoning", "thinking"),
        "coding": ("coding", "code generation", "programming"),
    }.get(feature, (feature,))


def _feature_status(feature: str, model_id: str, sources: list[dict[str, Any]]) -> str:
    target_url = f"https://huggingface.co/{model_id}".lower()
    for source in sources:
        body = str(source.get("retrieved_text") or "").lower()
        if not any(term in body for term in _feature_terms(feature)):
            continue
        url = str(source.get("source_url") or "").lower()
        if url.startswith(target_url) and (_hf_repository_url(url) or "").lower() == target_url:
            return "verified"
    for source in sources:
        body = str(source.get("retrieved_text") or "").lower()
        hostname = (urlparse(str(source.get("source_url") or "")).hostname or "").lower()
        credible = hostname in {"huggingface.co", "github.com"} or hostname.startswith(
            ("docs.", "developer.")
        )
        if (
            credible
            and _candidate_source(source, model_id)
            and any(term in body for term in _feature_terms(feature))
        ):
            return "inferred"
    return "unknown"


def discover_models(
    spec: RequestSpec,
    settings: dict[str, Any],
    artifact_dir: Path,
    *,
    scan: Callable[..., list[dict[str, Any]]] = scan_models,
    lookup: Callable[..., list[dict[str, Any]]] = lookup_models,
    search: Callable[..., list[dict[str, Any]]] = tavily_research,
    scan_config: dict[str, Any] | None = None,
    scoring_config: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    constraints = extract_constraints(spec.request)
    scan_config = dict(scan_config or load_yaml_file("config/scan.yaml"))
    scoring_config = scoring_config or load_yaml_file("config/scoring.yaml")
    scan_config["lookback_days"] = max(int(scan_config.get("lookback_days", 2)), 30)
    scan_config["min_downloads"] = 0
    if spec.license.lower() == "any":
        scan_config["allowed_licenses"] = []
    else:
        scan_config["allowed_licenses"] = [part.strip() for part in spec.license.split(",")]
    if constraints["target_params_billion"]:
        scan_config["max_params_billion"] = max(
            float(scan_config.get("max_params_billion", 30)),
            constraints["target_params_billion"] * 2,
        )
    sources: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    if progress:
        progress("Scanning recent Hugging Face models...")
    recent = scan(scan_config, artifact_dir / "recent.jsonl")
    scans = [("recent_similar", recent)]
    if progress:
        progress(f"Recent scan returned {len(recent)} models.")
    if constraints["family"] and constraints["family"].lower() != "gemma":
        if progress:
            progress(f"Searching {constraints['family']} family...")
        scans.append(
            (
                "same_family",
                lookup(
                    family=constraints["family"],
                    limit=25,
                    output_path=artifact_dir / "family.jsonl",
                ),
            )
        )
    if progress:
        progress("Searching Gemma family...")
    scans.append(
        ("gemma", lookup(family="Gemma", limit=25, output_path=artifact_dir / "gemma.jsonl"))
    )
    if progress:
        progress("Scoring and filtering Hub candidates...")
        for strategy, rows in scans:
            progress(f"{strategy}: {len(rows)} models found.")
    for strategy, rows in scans:
        diagnostics[strategy] = {
            "found": len(rows),
            "size_matched": 0,
            "eligible": 0,
            "blockers": {},
        }
        for row in rows:
            if not _approximately_sized(row, constraints["target_params_billion"]):
                continue
            diagnostics[strategy]["size_matched"] += 1
            ident = str(row.get("model_id") or "")
            if not ident:
                continue
            if ident not in sources:
                sources[ident] = {"record": row, "strategies": []}
            sources[ident]["strategies"].append(strategy)
            diagnosis = score_candidate(row, scan_config, scoring_config)
            if diagnosis["eligible"]:
                diagnostics[strategy]["eligible"] += 1
            else:
                for reason in diagnosis["blocking_reasons"]:
                    counts = diagnostics[strategy]["blockers"]
                    counts[reason] = counts.get(reason, 0) + 1
    candidates = []
    for entry in sources.values():
        row = entry["record"]
        scored = score_candidate(row, scan_config, scoring_config)
        if not scored["eligible"]:
            continue
        params = row.get("estimated_params_billion")
        footprint = row.get("estimated_download_size_gb") or (float(params) * 2 if params else None)
        if footprint is None or float(footprint) + 16 > spec.gpu_memory_gb * float(
            settings["selection"]["memory_safety_margin"]
        ):
            continue
        candidates.append({**scored, "record": row, "discovery_strategies": entry["strategies"]})
    candidates.sort(
        key=lambda item: (item["score"], len(item["discovery_strategies"])), reverse=True
    )
    # Preserve variety while keeping the strongest HF scored models.
    selected = []
    for strategy in ("recent_similar", "same_family", "gemma"):
        match = next(
            (
                item
                for item in candidates
                if strategy in item["discovery_strategies"] and item not in selected
            ),
            None,
        )
        if match and len(selected) < 3:
            selected.append(match)
    for item in candidates:
        if len(selected) >= 3:
            break
        if item not in selected:
            selected.append(item)
    if progress:
        progress(f"Selected {len(selected)} candidates for task-fit research.")
    evidence: dict[str, list[dict[str, Any]]] = {}
    for item in selected:
        ident = item["model_id"]
        query = (
            f"Candidate model: {ident}\nUser requirement: {spec.request}\n"
            "Evidence for requested capabilities, features, hardware, memory, "
            "inference speed, framework and deployment fit."
        )
        if progress:
            progress(f"Tavily Search: {ident}...")
        try:
            found = search(query, settings)
        except Exception as exc:
            if progress:
                progress(f"Tavily Search failed for {ident}: {type(exc).__name__}")
            found = []
        relevant = [source for source in found if _candidate_source(source, ident)]
        if progress:
            progress(
                f"Tavily Search returned {len(found)} sources for {ident}; "
                f"{len(relevant)} name this candidate."
            )
        evidence[ident] = relevant
        urls = [
            row["source_url"]
            for row in sorted(
                relevant,
                key=lambda row: (
                    f"huggingface.co/{ident.lower()}" in str(row.get("source_url", "")).lower(),
                    "github.com" in str(row.get("source_url", "")),
                    "huggingface.co" in str(row.get("source_url", "")),
                    float(row.get("score") or 0),
                ),
                reverse=True,
            )
            if str(row.get("source_url", "")).startswith("https://")
        ][:2]
        if urls:
            if progress:
                progress(f"Tavily Extract: {len(urls)} pages for {ident}...")
            try:
                extracted = search(query, settings, extract_urls=urls)
                extracted = [row for row in extracted if _candidate_source(row, ident)]
                for row in extracted:
                    row["extracted"] = True
                evidence[ident].extend(extracted)
                if progress:
                    progress(f"Tavily Extract returned {len(extracted)} pages for {ident}.")
            except Exception as exc:
                if progress:
                    progress(f"Tavily Extract failed for {ident}: {type(exc).__name__}")
        for feature in constraints["features"]:
            # The original request selects the features. Focus each follow-up on
            # this candidate so a reference model does not dominate results.
            terms = (
                "speculative decoding draft target model"
                if feature == "speculative decoding"
                else f"{feature} support documentation"
            )
            feature_query = f'"{ident}" {terms} site:huggingface.co OR site:github.com'
            if progress:
                progress(f"Tavily {feature} research: {ident}...")
            try:
                feature_results = search(feature_query, settings)
            except Exception as exc:
                feature_results = []
                if progress:
                    progress(f"{feature} search failed for {ident}: {type(exc).__name__}")
            relevant_feature = [
                source for source in feature_results if _candidate_source(source, ident)
            ]
            extract_urls = []
            if feature == "speculative decoding":
                # A draft may live in a different HF repository. Extract its
                # canonical model page, then use the existing exact-pair verifier.
                for source in feature_results:
                    repository_url = _hf_repository_url(str(source.get("source_url") or ""))
                    if (
                        repository_url
                        and repository_url.lower() != f"https://huggingface.co/{ident}".lower()
                    ):
                        if not any(term in repository_url.lower() for term in ("gguf", "gptq")):
                            extract_urls.append(repository_url)
                stem = ident.rsplit("/", 1)[-1].lower()
                extract_urls = sorted(
                    set(extract_urls), key=lambda url: stem in url.lower(), reverse=True
                )[:2]
            else:
                extract_urls = [
                    str(source["source_url"])
                    for source in relevant_feature
                    if str(source.get("source_url") or "").startswith("https://")
                ][:2]
            extracted_feature = []
            if extract_urls:
                if progress:
                    progress(f"Tavily Extract: {len(extract_urls)} {feature} pages for {ident}...")
                try:
                    extracted_feature = search(feature_query, settings, extract_urls=extract_urls)
                except Exception as exc:
                    if progress:
                        progress(f"{feature} extract failed for {ident}: {type(exc).__name__}")
            if feature == "speculative decoding":
                verified_pairs = [
                    source
                    for source in feature_results + extracted_feature
                    if _pair_evidence(item["record"], [source]) is not None
                ]
                relevant_feature.extend(verified_pairs)
                if progress:
                    progress(
                        f"Verified {len(verified_pairs)} direct speculative draft pair "
                        f"sources for {ident}."
                    )
            else:
                relevant_feature.extend(
                    source for source in extracted_feature if _candidate_source(source, ident)
                )
            for source in relevant_feature:
                source.setdefault("research_feature", feature)
            evidence[ident].extend(relevant_feature)
    if progress:
        progress("Comparing candidate fit and saving recommendations...")
    runtime = {
        row["model_id"]: row
        for row in rank_candidates([item["record"] for item in selected], spec, evidence, settings)
    }
    recommendations = []
    for item in selected:
        ident = item["model_id"]
        assessment = runtime[ident]
        blockers = list(assessment["blocking_reasons"])
        # The legacy runtime ranker defaults to a 64k document fixture for unrelated
        # requests. Discovery uses the requested generation length when available.
        context = assessment.get("context_length")
        minimum_context = (constraints["max_generation_tokens"] or 2048) + 2048
        if (
            "context_window_unverified_or_insufficient" in blockers
            and not any(term in spec.request.lower() for term in ("long context", "long document"))
            and context is not None
            and int(context) >= minimum_context
        ):
            blockers.remove("context_window_unverified_or_insufficient")
        if blockers:
            continue
        sources_for_model = evidence[ident]
        limitations = []
        if constraints["max_generation_tokens"]:
            limitations.append("Prompt length is unspecified; context fit is an estimate")
        if not sources_for_model:
            limitations.append("No Tavily evidence found for the task request")
        pair = assessment.get("speculative_decoding") or {}
        feature_statuses = {}
        for feature in constraints["features"]:
            status = _feature_status(feature, ident, sources_for_model)
            if feature == "speculative decoding":
                if pair.get("usable"):
                    status = "verified"
                    limitations.append("Documented external draft pair; A100 speedup is unmeasured")
                elif status == "verified":
                    # A mention on the target page does not establish a usable
                    # target/draft/runtime combination.
                    status = "inferred"
            feature_statuses[feature.replace(" ", "_")] = status
            if status == "unknown":
                limitations.append(f"{feature.capitalize()} support is unverified")
        feature_score = min(
            0.1,
            sum(
                0.05 if status == "verified" else 0.02 if status == "inferred" else 0
                for status in feature_statuses.values()
            ),
        )
        reasoning_parts = [
            f"The existing Hugging Face scanner marked {ident} eligible with a score "
            f"of {item['score']:.3f}."
        ]
        if constraints["target_params_billion"] is not None:
            reasoning_parts.append(
                f"Its estimated {item['record']['estimated_params_billion']:g}B parameters "
                f"are near the requested {constraints['target_params_billion']:g}B scale."
            )
        memory = assessment.get("estimated_bf16_memory_gb")
        if memory is not None:
            reasoning_parts.append(
                f"The estimated serving footprint is {memory:g} GB against "
                f"{spec.gpu_memory_gb:g} GB of GPU memory; this is a capacity estimate."
            )
        if spec.gpu_hourly_cost_usd is not None:
            reasoning_parts.append(
                f"The supplied GPU cost of ${spec.gpu_hourly_cost_usd:g}/hour "
                f"is within the ${spec.budget_usd_per_hour:g}/hour budget."
            )
        for feature, status in feature_statuses.items():
            name = feature.replace("_", " ")
            if feature == "speculative_decoding" and pair.get("usable"):
                reasoning_parts.append(
                    f"Speculative decoding is documented for this exact target with "
                    f"external draft {pair['draft_model']} via {pair['runtime']} "
                    f"({pair['source_url']}); speed on the requested GPU is unmeasured."
                )
            else:
                reasoning_parts.append(f"Evidence for {name} is {status}.")
        if limitations:
            reasoning_parts.append("Remaining uncertainty: " + "; ".join(limitations) + ".")
        recommendations.append(
            {
                "model_id": ident,
                "reasoning": " ".join(reasoning_parts),
                "hf_score": item["score"],
                "discovery_strategies": item["discovery_strategies"],
                "eligible": True,
                "final_score": round(
                    item["score"] + 0.1 + feature_score + (0.02 if sources_for_model else 0),
                    4,
                ),
                "why_it_fits": [
                    *(
                        [
                            "Documented external speculative draft: "
                            f"{pair['draft_model']} via {pair['runtime']}"
                        ]
                        if pair.get("usable")
                        else []
                    ),
                    *[
                        f"{feature.replace('_', ' ')} supported by candidate evidence"
                        for feature, status in feature_statuses.items()
                        if status == "verified" and feature != "speculative_decoding"
                    ],
                    f"HF scanner score {item['score']:.3f}",
                    f"Approximately {item['record'].get('estimated_params_billion')}B parameters",
                    f"Fits the stated {spec.gpu_memory_gb:g} GB memory estimate",
                ],
                "limitations": limitations,
                "task_fit": {
                    **feature_statuses,
                    "gpu_fit": "inferred"
                    if assessment["estimated_bf16_memory_gb"] is not None
                    else "unknown",
                    "framework_fit": "inferred"
                    if assessment["compatible_frameworks"]
                    else "unknown",
                },
                "evidence": sources_for_model,
                "runtime_assessment": assessment,
            }
        )
    recommendations.sort(key=lambda row: (not row["eligible"], -row["final_score"]))
    result = {
        "status": "recommended" if recommendations else "insufficient_evidence",
        "task_request": spec.request,
        "constraints": constraints,
        "scan_diagnostics": diagnostics,
        "recommendations": recommendations[:3],
        "selected_model_id": None,
        "action": "Optimize Inference + Benchmark",
        "action_status": "placeholder",
    }
    (artifact_dir / "discovery.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result
