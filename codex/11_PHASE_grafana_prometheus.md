# Codex Phase 11 — Prometheus and Grafana observability

Implement only Phase 11.

Assume Phases 1–10 are complete and tests pass.

Goal: add optional Prometheus/Grafana observability for operational and token-normalized benchmark metrics.

Do not replace SQLite/Postgres as the source of truth. Grafana is for visualization and monitoring only.

Implement:

- optional Prometheus metrics exporter
- metrics emitted during scan, worker, eval, fine-tune, upload, and orchestration
- example Grafana dashboard JSON
- documentation for running Grafana locally
- dry-run mode

Metrics to expose:

- hf_watch_models_scanned_total
- hf_watch_candidates_selected_total
- hf_watch_jobs_started_total
- hf_watch_jobs_completed_total
- hf_watch_jobs_failed_total
- hf_watch_failures_total{failure_category}
- hf_watch_model_load_seconds
- hf_watch_ttft_ms
- hf_watch_total_latency_ms
- hf_watch_decode_tokens_per_second
- hf_watch_input_tokens_total
- hf_watch_output_tokens_total
- hf_watch_total_tokens_total
- hf_watch_latency_per_1k_input_tokens
- hf_watch_latency_per_1k_output_tokens
- hf_watch_quality_score
- hf_watch_quality_per_1k_tokens
- hf_watch_artifact_upload_seconds
- hf_watch_artifact_upload_failures_total

Grafana dashboard panels:

- Runs over time
- Success/failure rate
- Failures by category
- TTFT by model/task
- Decode tokens/sec by model/task
- Latency per 1k input tokens
- Latency per 1k output tokens
- Total tokens by model/task
- Cost estimate over time
- Quality score by model/task
- Quality per 1k tokens
- GPU/model load time
- Artifact upload failures

Success criteria:

- Prometheus exporter runs locally
- tests pass
- dashboard JSON is created
- README explains how to run optional Grafana stack