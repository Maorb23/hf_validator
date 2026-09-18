from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from hf_model_watch.reporting.html_report import html_list, html_table, write_html_report
from hf_model_watch.reporting.markdown_report import bullet_list, json_block, markdown_table
from hf_model_watch.storage.sqlite_store import SQLiteStore

LEADERBOARD_COLUMNS = [
    "rank",
    "model_id",
    "revision_short",
    "family",
    "size_band",
    "scan_score",
    "eligibility_status",
    "selected_backend",
    "base_quality_score",
    "math_accuracy",
    "math_answer_latency_median_ms",
    "math_answer_latency_p95_ms",
    "sentence_embedding_similarity_mean",
    "sentence_embedding_similarity_p10",
    "sentence_reference_bits_per_byte_mean",
    "adapted_quality_score",
    "adaptation_lift",
    "normalized_adaptation_lift",
    "latency_tokens_sec",
    "ttft_ms",
    "context_drift_score",
    "peak_vram",
    "cost_estimate",
    "failure_category",
    "artifact_uri",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "total_cached_prompt_tokens",
    "total_reasoning_tokens",
    "total_cost_estimate_usd",
    "quality_per_1k_tokens",
    "cost_per_successful_task",
    "tokens_per_successful_task",
    "adaptation_lift_per_1M_training_tokens",
    "normalized_lift_per_1M_training_tokens",
    "usage_source_summary",
]


def generate_leaderboard_markdown(db_path: str | Path, out_path: str | Path) -> Path:
    outputs = generate_leaderboard_reports(db_path, out_path)
    return outputs["markdown"]


def generate_leaderboard_reports(db_path: str | Path, out_path: str | Path) -> dict[str, Path]:
    store = SQLiteStore(db_path)
    store.initialize()
    rows = build_leaderboard_rows(store.fetch_reporting_runs())
    markdown_path = Path(out_path)
    html_path = markdown_path.with_suffix(".html")
    reports_root = markdown_path.parent
    markdown = render_leaderboard(rows)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    write_html_report(
        "hf-model-watch Leaderboard",
        [
            (
                "Leaderboard",
                html_table([_title(column) for column in LEADERBOARD_COLUMNS], _row_values(rows)),
            )
        ],
        html_path,
    )
    for raw_run in store.fetch_reporting_runs():
        render_run_reports(raw_run, reports_root / "runs")
    return {"markdown": markdown_path, "html": html_path}


def build_leaderboard_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_build_row(run) for run in runs]
    rows.sort(
        key=lambda row: (
            row.get("adapted_quality_score")
            if row.get("adapted_quality_score") is not None
            else row.get("base_quality_score")
            if row.get("base_quality_score") is not None
            else -1.0,
            row.get("completed_at") or "",
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def render_leaderboard(rows: list[dict[str, Any]]) -> str:
    lines = ["# hf-model-watch Leaderboard", ""]
    if not rows:
        lines.extend(["No tested model results are available yet.", ""])
        return "\n".join(lines)
    lines.append(
        markdown_table([_title(column) for column in LEADERBOARD_COLUMNS], _row_values(rows))
    )
    lines.append("")
    return "\n".join(lines)


def render_run_reports(run: dict[str, Any], reports_dir: str | Path) -> dict[str, Path]:
    row = _build_row(run)
    run_id = str(run["run_id"])
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{run_id}.md"
    html_path = output_dir / f"{run_id}.html"
    sections = _run_report_sections(run, row)
    markdown = [f"# Run Report: {run_id}", ""]
    for heading, content in sections:
        markdown.extend([f"## {heading}", "", content, ""])
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    html_sections = []
    for heading, content in _run_report_html_sections(run, row):
        html_sections.append((heading, content))
    write_html_report(f"Run Report: {run_id}", html_sections, html_path)
    return {"markdown": markdown_path, "html": html_path}


def _build_row(run: dict[str, Any]) -> dict[str, Any]:
    latency = _loads(run.get("latency_summary"), {})
    context = _loads(run.get("context_drift_summary"), {})
    quality_payload = _payload(run.get("quality_results"))
    quality = (
        quality_payload.get("summary", {})
        if isinstance(quality_payload, dict)
        else {}
    )
    adaptation = _adaptation_payload(run)
    overall = adaptation.get("overall") if isinstance(adaptation.get("overall"), dict) else {}
    if not overall:
        task_results = adaptation.get("task_results") or []
        if task_results:
            overall = task_results[0]
    total_tokens = latency.get("total_tokens")
    successful_task_count = latency.get("successful_task_count")
    quality_score = run.get("base_score")
    quality_per_1k = latency.get("avg_quality_per_1k_tokens")
    if quality_per_1k is None and quality_score is not None and total_tokens is not None:
        quality_per_1k = 1000 * float(quality_score) / max(int(total_tokens), 1)
    total_cost = latency.get("total_cost_estimate_usd")
    return {
        "run_id": run.get("run_id"),
        "completed_at": run.get("completed_at"),
        "model_id": run.get("model_id"),
        "revision_short": _short_revision(run.get("revision")),
        "family": run.get("model_family"),
        "size_band": run.get("size_band"),
        "scan_score": run.get("scan_score"),
        "eligibility_status": _eligibility_status(run),
        "selected_backend": run.get("selected_backend"),
        "base_quality_score": run.get("base_score"),
        "math_accuracy": quality.get("math_accuracy"),
        "math_answer_latency_median_ms": quality.get("math_answer_latency_median_ms"),
        "math_answer_latency_p95_ms": quality.get("math_answer_latency_p95_ms"),
        "sentence_embedding_similarity_mean": quality.get(
            "sentence_embedding_similarity_mean"
        ),
        "sentence_embedding_similarity_p10": quality.get(
            "sentence_embedding_similarity_p10"
        ),
        "sentence_reference_bits_per_byte_mean": quality.get(
            "sentence_reference_bits_per_byte_mean"
        ),
        "adapted_quality_score": run.get("adapted_score"),
        "adaptation_lift": run.get("adaptation_lift"),
        "normalized_adaptation_lift": overall.get("normalized_adaptation_lift"),
        "latency_tokens_sec": latency.get("avg_decode_tokens_per_second"),
        "ttft_ms": latency.get("avg_ttft_ms"),
        "context_drift_score": context.get("drift_score"),
        "peak_vram": latency.get("peak_vram") or latency.get("gpu_peak_memory"),
        "cost_estimate": total_cost,
        "failure_category": run.get("failure_category"),
        "artifact_uri": run.get("artifact_uri"),
        "total_prompt_tokens": latency.get("total_prompt_tokens"),
        "total_completion_tokens": latency.get("total_completion_tokens"),
        "total_tokens": total_tokens,
        "total_cached_prompt_tokens": latency.get("total_cached_prompt_tokens"),
        "total_reasoning_tokens": latency.get("total_reasoning_tokens"),
        "total_cost_estimate_usd": total_cost,
        "quality_per_1k_tokens": quality_per_1k,
        "cost_per_successful_task": _safe_div(total_cost, successful_task_count),
        "tokens_per_successful_task": _safe_div(total_tokens, successful_task_count),
        "adaptation_lift_per_1M_training_tokens": overall.get(
            "adaptation_lift_per_1M_training_tokens"
        ),
        "normalized_lift_per_1M_training_tokens": overall.get(
            "normalized_lift_per_1M_training_tokens"
        ),
        "usage_source_summary": _usage_source_summary(run),
    }


def _run_report_sections(run: dict[str, Any], row: dict[str, Any]) -> list[tuple[str, str]]:
    artifacts = run.get("artifacts") or {}
    adaptation = _adaptation_payload(run)
    sections = [
        ("Summary", bullet_list({"Run ID": run.get("run_id"), "Status": run.get("status")})),
        (
            "Model Metadata",
            bullet_list(
                {
                    "Model": run.get("model_id"),
                    "Revision": run.get("revision"),
                    "Family": run.get("model_family"),
                    "Size band": run.get("size_band"),
                }
            ),
        ),
        ("Runtime Resolution", bullet_list({"Backend": run.get("selected_backend")})),
        ("Claim Verification", json_block(_payload(run.get("metadata_results")))),
        ("Latency Results", json_block(_loads(run.get("latency_summary"), {}))),
        ("Latency Prompts", _eval_records_table(_records_from_payload(run.get("latency_results")))),
        ("Quality Results", json_block(_payload(run.get("quality_results")))),
        ("Quality Prompts", _eval_records_table(_records_from_payload(run.get("quality_results")))),
        ("Context Drift", json_block(_loads(run.get("context_drift_summary"), {}))),
        (
            "Context Drift Prompts",
            _eval_records_table(_records_from_payload(run.get("context_drift_results"))),
        ),
        (
            "Token Usage",
            bullet_list(
                {
                    "Input tokens": row.get("total_prompt_tokens"),
                    "Output tokens": row.get("total_completion_tokens"),
                    "Total tokens": row.get("total_tokens"),
                    "Cached input tokens": row.get("total_cached_prompt_tokens"),
                    "Reasoning tokens": row.get("total_reasoning_tokens"),
                    "Usage source": row.get("usage_source_summary"),
                }
            ),
        ),
        (
            "Performance",
            bullet_list(
                {
                    "Average total latency (ms)": _loads(
                        run.get("latency_summary"), {}
                    ).get("avg_total_latency_ms"),
                    "Average decode latency (ms)": _loads(
                        run.get("latency_summary"), {}
                    ).get("avg_decode_latency_ms"),
                    "Average TTFT (ms)": row.get("ttft_ms"),
                    "Decode tokens/sec": row.get("latency_tokens_sec"),
                }
            ),
        ),
        (
            "Token Efficiency",
            bullet_list(
                {
                    "Quality per 1k tokens": row.get("quality_per_1k_tokens"),
                    "Tokens per successful task": row.get("tokens_per_successful_task"),
                    "Cost per successful task": row.get("cost_per_successful_task"),
                }
            ),
        ),
        ("PEFT Config", json_block(_payload(run.get("finetune_runs")))),
        (
            "Adaptation Lift",
            bullet_list(
                {
                    "Base quality score": row.get("base_quality_score"),
                    "Adapted quality score": row.get("adapted_quality_score"),
                    "Raw adaptation lift": row.get("adaptation_lift"),
                    "Normalized adaptation lift": row.get("normalized_adaptation_lift"),
                }
            ),
        ),
        (
            "Adaptation Efficiency",
            bullet_list(
                {
                    "Training tokens": _training_tokens(adaptation),
                    "Adaptation lift per 1M training tokens": row.get(
                        "adaptation_lift_per_1M_training_tokens"
                    ),
                    "Normalized lift per 1M training tokens": row.get(
                        "normalized_lift_per_1M_training_tokens"
                    ),
                }
            ),
        ),
        (
            "Failures",
            bullet_list(
                {
                    "Failure category": run.get("failure_category"),
                    "Failure message": run.get("failure_message"),
                }
            ),
        ),
        ("Artifacts", bullet_list(artifacts)),
    ]
    return sections


def _run_report_html_sections(run: dict[str, Any], row: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (
            heading,
            f"<pre>{html.escape(content)}</pre>"
            if "```json" in content
            else html_list(_section_dict(content)),
        )
        for heading, content in _run_report_sections(run, row)
    ]


def _section_dict(content: str) -> dict[str, str]:
    items = {}
    for line in content.splitlines():
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            items[key] = value.strip()
    return items or {"value": content}


def _row_values(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [[row.get(column) for column in LEADERBOARD_COLUMNS] for row in rows]


def _title(column: str) -> str:
    return column.replace("_", " ").title()


def _short_revision(revision: Any) -> str:
    return "" if not revision else str(revision)[:12]


def _loads(raw: Any, default: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return default
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return default


def _payload(row: dict[str, Any] | None) -> Any:
    if not row:
        return {}
    for key in ("payload_json", "claims_vs_observed_json"):
        if row.get(key):
            return _loads(row[key], {})
    return {}


def _records_from_payload(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = _payload(row)
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
    return []


def _eval_records_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "N/A"
    rows = []
    for record in records:
        rows.append(
            [
                record.get("prompt_id"),
                record.get("task_type"),
                _truncate(record.get("prompt_text") or record.get("rendered_prompt_text")),
                _truncate(_jsonish(record.get("expected"))),
                _truncate(record.get("generated_text")),
                record.get("quality_score"),
                record.get("success"),
            ]
        )
    return markdown_table(
        ["Prompt ID", "Task", "Prompt", "Expected", "Output", "Score", "Success"],
        rows,
    )


def _jsonish(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _truncate(value: Any, limit: int = 240) -> str:
    if value is None:
        return "N/A"
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _adaptation_payload(run: dict[str, Any]) -> dict[str, Any]:
    return _payload(run.get("adaptation_results"))


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    return float(numerator) / max(int(denominator), 1)


def _eligibility_status(run: dict[str, Any]) -> str:
    if run.get("eligible") is None:
        return "unknown"
    if run.get("eligible"):
        return "eligible"
    return "blocked"


def _usage_source_summary(run: dict[str, Any]) -> str:
    sources: set[str] = set()
    for table in ("latency_results", "quality_results", "context_drift_results"):
        payload = _payload(run.get(table))
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for record in records:
            if record.get("usage_source"):
                sources.add(str(record["usage_source"]))
    if not sources:
        return "missing"
    return next(iter(sources)) if len(sources) == 1 else "mixed"


def _training_tokens(adaptation: dict[str, Any]) -> Any:
    overall = adaptation.get("overall") if isinstance(adaptation.get("overall"), dict) else {}
    if overall.get("training_total_tokens") is not None:
        return overall["training_total_tokens"]
    for result in adaptation.get("task_results") or []:
        if result.get("training_total_tokens") is not None:
            return result["training_total_tokens"]
    return None
