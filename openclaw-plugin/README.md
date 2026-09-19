# hf-model-watch OpenClaw plugin

This tool-only plugin exposes six restricted operations backed by the existing `hf-watch` CLI and
Phase 1 recommendation artifacts. It does not expose shell access or an optimization operation.

Prerequisites: OpenClaw 2026.5.17 or newer, Node 24.16+/26.1+, and the project's lightweight Python
environment. Configure `projectRoot` to this repository and, when needed, `pythonExecutable` to
`.venv/bin/python`. `stateRoot` defaults to `artifacts/recommendations`.

Build and validate with:

```console
npm install
npm run plugin:validate
openclaw plugins install .
```

The plugin launches only `python -m hf_model_watch.openclaw_adapter`, with JSON over stdin/stdout,
an argument-array subprocess, bounded output, and a timeout. The Python bridge then invokes only the
allowlisted CLI operations with validated arguments.
