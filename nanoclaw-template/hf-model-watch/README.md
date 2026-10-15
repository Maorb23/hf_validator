# hf-model-watch NanoClaw template

This is a NanoClaw Agent Plugin template. It provides recommendation and optional optimization MCP tools backed by the existing
Python restricted adapter: recommendation, clarification, selection, feedback, revision, and run
status, plus optional optimization start, status, prior-study listing, reuse, and cancel.
It does not provide shell, filesystem, Docker, or credential
operations.

The `bin/hf-model-watch-mcp` launcher expects the staged runtime at
`/workspace/extra/hf-model-watch/runtime`. Follow `docs/nanoclaw.md` to create that runtime and
mount it along with the separate writable recommendation state directory.
