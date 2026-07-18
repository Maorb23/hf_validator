# Reference Scripts

These scripts are retained as architectural references for later serverless phases.
Phase 1 copies only broad conventions:

- `set -euo pipefail`
- stdout lines prefixed with `TRACE_EVENT `
- env-driven configuration
- explicit bootstrap events
- fail-fast shell behavior

The autoresearch-specific training, metrics, repositories, and datasets are not part of
`hf-model-watch`.
