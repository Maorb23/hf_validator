# Pinned optimizer source

`llm_optimizer-0.0.1-915beb35.tar.gz` is a source distribution input from
`Maorb23/llm-optimizer-trt`, commit
`915beb35e4b2b398f1009c9e22d0fd3da3243d29` (package version 0.0.1).
It includes only `src`, `pyproject.toml`, `README.md`, and the MIT `LICENSE`.
No results, databases, keys, local edits, or Git history are included.

The fork is not anonymously downloadable. Keeping this reviewed, checksummed
snapshot in this repository lets Railway build without GitHub credentials or
an adjacent checkout. Installation uses ordinary pip package metadata and CPU
platform dependencies; no serving runtime or torch is installed.

To upgrade, export the same four paths from a reviewed full Git commit with
`git archive --format=tar.gz --prefix=llm-optimizer-<short-sha>/`, update the
requirement filename and SHA256SUMS (paths relative to the repository root),
and rerun the platform/deployment checks in an isolated environment. Never
archive the whole working tree. Retain the license inside the archive.
