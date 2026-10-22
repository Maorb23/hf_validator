# NanoClaw runtime update, 2026-10-18

Updated the installed HF Model Watch Python runtime, its llm-optimizer package, the
NanoClaw agent plugin instructions, the template, and the runtime documentation.
The existing agent image and MCP/OneCLI registration were retained. The HF Model Watch
group was restarted; it loads the new instructions/runtime on its next user message.

Included behavior:

- Required workload/optimization intake before recommendation research, with saved preferences.
- Automatic intent/request preparation after model selection and numeric input normalization.
- Shared expanded search spaces for vLLM, SGLang and TRT-LLM, including concurrency 32.
- TRT config-file materialization and compressed-payload support in the installed optimizer.
- $1 per GPU-hour planning default; explicit/saved prices override it.
- User-provided default image references and GPU-specific presets, plus the existing named
  spot policy. Image names do not establish verified worker capabilities.
- Dry-run versus live preparation instructions, followed by separate explicit submission approval.

Operator settings remain dry-run, with a 60-minute timeout and 16 trials. TPESampler is
explicit. No credentials or `.env` were copied into the runtime; only the existing non-secret
spot policy ID was added to execution configuration. Conversation state and optimizer run/job
counts were verified unchanged. No study was submitted, no reconciliation loop was started,
and no GPU job was launched.

The previous build is saved at:

`/home/maorb/hf-model-watch-nanoclaw/backups/before-expanded-optimization-20261018T100624Z/`

It includes the full prior runtime, agent directory (including state), template, effective
container configuration, mount allowlist, consistent NanoClaw/optimizer SQLite snapshots,
previous image ID and retained image tag, and rollback notes. The backup directory is private.

Verification: 66 focused Python tests passed. Isolated Docker tests passed before deployment
and against the installed Python 3.11 runtime afterward, with networking disabled and temporary
state. The installed test exercised stdio MCP tool listing, selection and preparation, expanded
searches for all three frameworks, TRT config files, preserved limits and rejection of submission
without approval. This does not establish live model, provider, or GPU benchmark behavior.

After the operator reported rebuilding and pushing all three GPU-serving images, CLI defaults
and the installed NanoClaw execution configuration were updated to `expanded-v1`. The prior
NanoClaw configuration and installed intake module were saved under
`/home/maorb/hf-model-watch-nanoclaw/backups/before-expanded-v1-tags-20261018T115106Z/`.
Existing prepared requests retain their original images and must be regenerated to use the new
tags. Image publication does not establish successful live GPU serving. See
[optimization.md](optimization.md).
