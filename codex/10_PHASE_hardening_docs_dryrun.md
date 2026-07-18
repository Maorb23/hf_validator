# Codex Phase 10 — final hardening, docs, end-to-end dry run

Implement only Phase 10.

Assume Phases 1–9 are complete and tests pass.

Goal: harden the full project, improve docs, run end-to-end dry runs, and remove placeholders.

Tasks:

1. Search for placeholders/TODOs.

   * Remove placeholder functions unless explicitly documented as future optional integrations.
   * Any remaining TODO must be listed in README under known limitations.

2. Harden failure behavior.

   * model-level failures should never crash the entire pipeline
   * infrastructure failures may exit nonzero
   * every failure should have a typed category

3. Harden reproducibility.

   * every run should store:

     * model_id
     * revision
     * config snapshot
     * package versions
     * CUDA availability
     * GPU name if available
     * selected backend
     * selected dtype
     * selected quantization
     * random seed
     * artifact URIs

4. Update README with:

   * what the project does
   * architecture
   * local setup
   * local tiny model run
   * scanner/scoring run
   * worker dry run
   * Prefect dry run
   * Nebius service account setup
   * S3/object storage setup
   * result storage
   * reports and leaderboard
   * known limitations

5. Add `.env.example` with all relevant env vars.

6. Run full local dry-run sequence:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run hf-watch scan --config config/scan.yaml
uv run hf-watch score-candidates --input artifacts/scan/latest_models.jsonl
uv run hf-watch run-local --model-id sshleifer/tiny-gpt2 --config-dir config --skip-finetune
uv run hf-watch leaderboard --db results/hf_model_watch.sqlite --out reports/leaderboard.md
uv run hf-watch compare --model-id sshleifer/tiny-gpt2 --db results/hf_model_watch.sqlite
uv run hf-watch prefect-flow --config-dir config --dry-run
DRY_RUN=1 bash scripts/run_model_worker_job.sh
DRY_RUN=1 bash scripts/run_prefect_controller_job.sh
```

Final required outputs:

```text
results/hf_model_watch.sqlite
artifacts/<run_id>/run_manifest.json
artifacts/<run_id>/metadata/claims_vs_observed.json
artifacts/<run_id>/latency_results.jsonl
artifacts/<run_id>/quality_results.jsonl
artifacts/<run_id>/context_drift_results.jsonl
artifacts/<run_id>/trace_events.jsonl
reports/leaderboard.md
reports/leaderboard.html
```

Final success criteria:

* all tests pass
* local tiny model run succeeds
* dry-run worker succeeds
* dry-run Prefect flow succeeds
* dry-run controller script succeeds
* no undocumented placeholders remain
* README is enough for a new engineer to run the project

At the end, report:

* final file tree
* commands run
* test status
* generated artifacts
* known limitations
* next recommended production steps
