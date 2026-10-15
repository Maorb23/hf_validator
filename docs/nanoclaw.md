# NanoClaw prototype

## Architecture

NanoClaw's supported integration surface is a template-owned stdio MCP server. The
`nanoclaw-template/hf-model-watch` template starts `hf_model_watch.nanoclaw_mcp` inside the
isolated agent container. That server exposes recommendation and optional optimization MCP tools and delegates them to the existing
`OpenClawToolAdapter`, which remains the validation and state-transition boundary over the existing
CLI. The adapter name is retained for compatibility; it is not specific to the MCP transport.

```text
NanoClaw local CLI -> agent container -> template stdio MCP server
                                      -> existing restricted Python adapter -> existing CLI
                                      -> recommendation state mount
```

The agent does not receive the host repository, Docker socket, or host credentials. The only extra
mounts are an installed, read-only Python runtime and a writable recommendation-state directory.
NanoClaw's template format does not allow a template to add mounts; the operator must make the two
explicit, allowlisted mount decisions below.

## WSL setup

A Docker daemon reachable from the WSL distribution is required before a live NanoClaw session can
run. Verify it from WSL:

```bash
docker version
```

Install NanoClaw and this repository separately in Linux-native paths (for example,
`/home/maorb/nanoclaw-hf-model-watch` and `/home/maorb/hf-validator`):

```bash
cd /home/maorb/nanoclaw-hf-model-watch
bash nanoclaw.sh
```

The setup wizard supports NanoClaw's local CLI channel. Complete provider authentication there;
OpenClaw credentials are not reused automatically.

Stage the minimal application runtime and create separate state, without putting either under
`/mnt/c`:

```bash
mkdir -p /home/maorb/hf-model-watch-nanoclaw/state
cd /home/maorb/hf-validator
bash scripts/prepare_nanoclaw_runtime.sh /home/maorb/hf-model-watch-nanoclaw/runtime
cp -R nanoclaw-template/hf-model-watch /home/maorb/nanoclaw-hf-model-watch/templates/
```

Create the template agent and note its returned group ID:

```bash
cd /home/maorb/nanoclaw-hf-model-watch
ncl groups create --template hf-model-watch --name "HF Model Watch"
```

For that group ID, configure the Python runtime and the two deliberately narrow mounts, then restart:

```bash
ncl groups config add-package --id <group-id> --apt python3-venv
ncl groups config add-mount --id <group-id> \
  --host /home/maorb/hf-model-watch-nanoclaw/runtime \
  --container /workspace/extra/hf-model-watch/runtime --ro
ncl groups config add-mount --id <group-id> \
  --host /home/maorb/hf-model-watch-nanoclaw/state \
  --container /workspace/extra/hf-model-watch/state --rw
ncl groups restart --id <group-id> --rebuild
```

Use NanoClaw's configured local CLI chat surface to describe a workload. The agent should call
`recommend_models`, ask for any returned clarifications, present the supplied shortlist, require
explicit selection, record feedback, and offer a justified revision. The `Optimize Inference +
Benchmark` action must remain unavailable.

## Security and limits

This is a single-user local prototype. The installed runtime is read-only and deliberately omits
`.env`, source checkout metadata, artifacts, and Docker access. The state mount is writable because
the existing CLI persists runs, selections, feedback, and revisions there. NanoClaw's default agent
container still includes its own container-local tools, so this is not a public multi-tenant design.

Live conversation validation requires Docker plus a provider credential. The Python MCP contract can
be validated without either using the test suite below.

Optional inference optimization setup and contracts are documented in [optimization.md](optimization.md).
