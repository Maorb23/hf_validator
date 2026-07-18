# Codex Phase 2 — Hugging Face scanner, filters, scoring, lineage

Implement only Phase 2.

Assume Phase 1 is complete and tests pass.

Goal: implement Hugging Face candidate discovery, blocking filters, scoring, and lineage extraction. Do not implement model loading, latency benchmarks, fine-tuning, Nebius, or Prefect yet.

Create/modify:

```text
src/hf_model_watch/hub/scan.py
src/hf_model_watch/hub/metadata.py
src/hf_model_watch/hub/candidate_filter.py
src/hf_model_watch/hub/scoring.py
src/hf_model_watch/hub/lineage.py

tests/test_candidate_filter.py
tests/test_scoring.py
tests/test_lineage.py
```

Implement `hub/scan.py`:

* use `huggingface_hub`
* discover recent/popular models
* support config values:

  * `lookback_days`
  * `max_models_to_scan`
  * `max_candidates_to_run`
  * `pipeline_tags`
  * `max_params_billion`
  * `allow_gated`
  * `allow_trust_remote_code`
  * `require_safetensors`
  * `max_download_gb`
* collect:

  * model_id
  * author/org
  * last_modified
  * downloads
  * likes
  * pipeline_tag
  * tags
  * license
  * gated/private status if available
  * sibling files
  * config metadata when available
  * tokenizer metadata when available
  * commit SHA / revision when available
  * safetensors presence
  * estimated parameter count when available
  * estimated download size when available
  * architecture from config
  * max_position_embeddings/context hints
  * tokenizer chat template presence

Output:

```text
artifacts/scan/latest_models.jsonl
```

Implement `candidate_filter.py`.

Blocking skip categories:

```text
not_causal_lm
too_large_for_h100
missing_tokenizer
missing_config
missing_weights
missing_safetensors
gated_no_access
license_blocked
already_tested_revision
requires_trust_remote_code
unsupported_transformers_architecture
download_too_large
```

Rules:

* block non-causal-LM candidates
* block missing config/tokenizer/weights
* block missing safetensors when required
* block gated/private when no access
* block disallowed license
* block above max size
* block above max download size
* block same revision if already tested
* block if `trust_remote_code` seems required while disabled
* return structured result, not exception

Implement `scoring.py`.

Eligible candidates get a score using configurable weights from `config/scoring.yaml`.

Use features:

* recency
* downloads
* likes
* credible org/author allowlist
* complete model card
* chat template present
* context length claim present
* predecessor family detectable
* known family
* useful size band
* not previously tested
* not likely duplicate/reupload

Return:

```json
{
  "model_id": "org/model",
  "revision": "abc123",
  "eligible": true,
  "score": 0.83,
  "blocking_reasons": [],
  "score_breakdown": {}
}
```

Implement `lineage.py`.

The lineage module should:

* normalize model IDs
* extract family name where obvious
* extract size where obvious
* extract version/date where obvious
* group by family and size band
* never claim predecessor performance unless local DB later has tested results
* return `inconclusive` when insufficient data

Families to recognize heuristically:

```text
Qwen
Llama
Mistral
Gemma
Phi
DeepSeek
Yi
```

Wire CLI:

```bash
hf-watch scan --config config/scan.yaml
hf-watch score-candidates --input artifacts/scan/latest_models.jsonl
```

The scanner should write JSONL. The scoring command should write:

```text
artifacts/scan/scored_candidates.jsonl
```

Tests:

* candidate filter blocks missing tokenizer
* candidate filter blocks too-large model
* candidate filter blocks gated model when not allowed
* candidate filter allows valid mock causal LM
* scoring ranks a newer/popular/complete candidate above a weak one
* lineage extracts family/size from obvious names
* lineage returns inconclusive when no prior tested models are supplied

Success criteria:

```bash
uv run ruff check .
uv run pytest
uv run hf-watch scan --config config/scan.yaml
uv run hf-watch score-candidates --input artifacts/scan/latest_models.jsonl
```

The commands must create:

```text
artifacts/scan/latest_models.jsonl
artifacts/scan/scored_candidates.jsonl
```

Stop after Phase 2. Report files created/modified, commands run, test results, and any TODOs.
