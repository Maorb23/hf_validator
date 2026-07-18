from __future__ import annotations

from pathlib import Path
from typing import Any

from hf_model_watch.reporting.leaderboard import build_leaderboard_rows
from hf_model_watch.reporting.markdown_report import bullet_list, markdown_table
from hf_model_watch.storage.sqlite_store import SQLiteStore

DEFAULT_THRESHOLDS = {
    "min_quality_delta": 0.02,
    "min_context_drift_delta": 0.02,
    "min_latency_delta_pct": 0.05,
    "min_token_efficiency_delta_pct": 0.05,
    "quality_comparable_margin": 0.01,
}


def compare_model(
    db_path: str | Path,
    model_id: str,
    out_dir: str | Path = "reports/comparisons",
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    store = SQLiteStore(db_path)
    store.initialize()
    current = _latest_tested_model_run(store, model_id)
    output_path = Path(out_dir) / f"{slugify_model_id(model_id)}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if current is None:
        result = {
            "model_id": model_id,
            "comparison_status": "inconclusive",
            "reason": "model has no tested runs in the local database",
            "report_path": str(output_path),
            "evidence": [],
        }
        output_path.write_text(render_comparison_report(result), encoding="utf-8")
        return result

    pool = store.fetch_tested_comparison_pool(str(current["run_id"]))
    result = compare_run_to_prior(current, pool, thresholds)
    result["report_path"] = str(output_path)
    output_path.write_text(render_comparison_report(result), encoding="utf-8")
    store.record_comparison_result(
        run_id=str(current["run_id"]),
        model_id=str(current["model_id"]),
        revision=current.get("revision"),
        compared_to_model_id=result.get("best_prior", {}).get("model_id"),
        compared_to_revision=result.get("best_prior", {}).get("revision"),
        payload=result,
    )
    return result


def compare_run_to_prior(
    current_run: dict[str, Any],
    prior_runs: list[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    config = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    current = build_leaderboard_rows([current_run])[0]
    candidates = _same_family_size(current_run, prior_runs)
    if not candidates:
        return _inconclusive(current, "no prior tested models in the same family and size band")
    prior_rows = build_leaderboard_rows(candidates)
    best_prior = max(prior_rows, key=lambda row: _quality(row) or -1.0)
    metrics = _metric_evidence(current, best_prior)
    status = _status_from_metrics(current, best_prior, metrics, config)
    return {
        "model_id": current["model_id"],
        "run_id": current["run_id"],
        "revision": current_run.get("revision"),
        "comparison_status": status,
        "reason": _reason_for_status(status, metrics),
        "current": current,
        "best_prior": best_prior,
        "evidence": metrics,
    }


def render_comparison_report(result: dict[str, Any]) -> str:
    lines = [
        f"# Comparison: {result.get('model_id')}",
        "",
        bullet_list(
            {
                "Status": result.get("comparison_status"),
                "Reason": result.get("reason"),
            }
        ),
        "",
    ]
    current = result.get("current") or {}
    prior = result.get("best_prior") or {}
    if not current or not prior:
        lines.append("No compact comparison table is available.")
        lines.append("")
        return "\n".join(lines)
    headers = ["Metric", "Current model", "Best prior same-family model", "Winner"]
    rows = [
        [item["metric"], item["current"], item["prior"], item["winner"]]
        for item in result.get("evidence", [])
    ]
    lines.append(markdown_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "-").replace(":", "-")


def _latest_tested_model_run(store: SQLiteStore, model_id: str) -> dict[str, Any] | None:
    runs = store.fetch_tested_runs_for_model(model_id)
    if not runs:
        return None
    completed = [
        run for run in runs if run.get("status") == "completed" or run.get("base_score") is not None
    ]
    return completed[0] if completed else runs[0]


def _same_family_size(
    current_run: dict[str, Any],
    prior_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    family = current_run.get("model_family")
    size = current_run.get("size_band")
    return [
        run
        for run in prior_runs
        if run.get("model_family") == family
        and run.get("size_band") == size
        and (run.get("status") == "completed" or run.get("base_score") is not None)
    ]


def _metric_evidence(current: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _metric_row("quality_score", _quality(current), _quality(prior), higher_is_better=True),
        _metric_row(
            "context_drift_score",
            current.get("context_drift_score"),
            prior.get("context_drift_score"),
            higher_is_better=False,
        ),
        _metric_row(
            "latency_per_1k_output_tokens",
            current.get("latency_per_1k_output_tokens"),
            prior.get("latency_per_1k_output_tokens"),
            higher_is_better=False,
        ),
        _metric_row(
            "quality_per_1k_tokens",
            current.get("quality_per_1k_tokens"),
            prior.get("quality_per_1k_tokens"),
            higher_is_better=True,
        ),
        _metric_row(
            "cost_per_successful_task",
            current.get("cost_per_successful_task"),
            prior.get("cost_per_successful_task"),
            higher_is_better=False,
        ),
        _metric_row(
            "adaptation_lift",
            current.get("adaptation_lift"),
            prior.get("adaptation_lift"),
            higher_is_better=True,
        ),
    ]


def _status_from_metrics(
    current: dict[str, Any],
    prior: dict[str, Any],
    metrics: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> str:
    quality_delta = _delta(_quality(current), _quality(prior))
    drift_delta = _delta(prior.get("context_drift_score"), current.get("context_drift_score"))
    latency_delta_pct = _pct_delta(
        prior.get("latency_per_1k_output_tokens"),
        current.get("latency_per_1k_output_tokens"),
    )
    token_delta_pct = _pct_delta(
        current.get("quality_per_1k_tokens"),
        prior.get("quality_per_1k_tokens"),
    )
    if _usage_small_estimated_mismatch(current, prior, token_delta_pct, thresholds):
        return "inconclusive"
    if quality_delta is not None and quality_delta < -thresholds["min_quality_delta"]:
        return "newer_worse"
    if drift_delta is not None and drift_delta < -thresholds["min_context_drift_delta"]:
        return "newer_worse"
    quality_better = quality_delta is not None and quality_delta >= thresholds["min_quality_delta"]
    quality_similar = quality_delta is not None and abs(quality_delta) <= thresholds[
        "quality_comparable_margin"
    ]
    drift_better_or_similar = drift_delta is None or drift_delta >= -thresholds[
        "min_context_drift_delta"
    ]
    latency_better_or_similar = latency_delta_pct is None or latency_delta_pct >= -thresholds[
        "min_latency_delta_pct"
    ]
    latency_better = latency_delta_pct is not None and latency_delta_pct >= thresholds[
        "min_latency_delta_pct"
    ]
    token_better = token_delta_pct is not None and token_delta_pct >= thresholds[
        "min_token_efficiency_delta_pct"
    ]
    if quality_better and drift_better_or_similar and latency_better_or_similar:
        context_better = (drift_delta or 0.0) >= thresholds["min_context_drift_delta"]
        if latency_better or token_better or context_better:
            return "newer_better"
    if quality_similar and drift_better_or_similar and latency_better_or_similar:
        return "similar"
    if quality_better:
        return "mixed"
    if any(item["winner"] == "current" for item in metrics) and any(
        item["winner"] == "prior" for item in metrics
    ):
        return "mixed"
    return "similar"


def _inconclusive(current: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "model_id": current["model_id"],
        "run_id": current["run_id"],
        "comparison_status": "inconclusive",
        "reason": reason,
        "current": current,
        "best_prior": {},
        "evidence": [],
    }


def _reason_for_status(status: str, metrics: list[dict[str, Any]]) -> str:
    if status == "inconclusive":
        return "insufficient comparable stored metrics"
    winners = {item["winner"] for item in metrics}
    if status == "mixed":
        return "quality improved but at least one efficiency or latency metric worsened"
    if status == "newer_better":
        return "quality is comparable or better with supporting latency/context evidence"
    if status == "newer_worse":
        return "quality or context drift is meaningfully worse"
    if winners <= {"tie", "missing"}:
        return "differences are below configured thresholds"
    return "differences are small enough to treat as similar"


def _metric_row(
    metric: str,
    current: Any,
    prior: Any,
    higher_is_better: bool,
) -> dict[str, Any]:
    winner = "missing"
    if current is not None and prior is not None:
        if float(current) == float(prior):
            winner = "tie"
        elif (float(current) > float(prior)) == higher_is_better:
            winner = "current"
        else:
            winner = "prior"
    return {"metric": metric, "current": current, "prior": prior, "winner": winner}


def _quality(row: dict[str, Any]) -> float | None:
    value = row.get("adapted_quality_score")
    if value is None:
        value = row.get("base_quality_score")
    return None if value is None else float(value)


def _delta(current: Any, prior: Any) -> float | None:
    if current is None or prior is None:
        return None
    return float(current) - float(prior)


def _pct_delta(current: Any, prior: Any) -> float | None:
    if current is None or prior in (None, 0):
        return None
    return (float(current) - float(prior)) / abs(float(prior))


def _usage_small_estimated_mismatch(
    current: dict[str, Any],
    prior: dict[str, Any],
    token_delta_pct: float | None,
    thresholds: dict[str, float],
) -> bool:
    if token_delta_pct is None:
        return False
    sources = {current.get("usage_source_summary"), prior.get("usage_source_summary")}
    mismatched = "estimated" in sources and "provider_reported" in sources
    return mismatched and abs(token_delta_pct) < thresholds["min_token_efficiency_delta_pct"]
