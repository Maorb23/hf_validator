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

Conversation memory belongs to one NanoClaw cluster and group. Give every
cluster/group its own host state directory and mount only that directory into its
agent container. Never share this mount with another group, cluster, or the
standalone CLI's artifact directory. The examples below provision one group;
choose a different host state path for each additional group.

Standalone CLI recommendations do not load conversation defaults, and CLI
optimization preparation does not read or update conversation memory. Explicitly
resuming a run preserves that run's settings. NanoClaw passes its group defaults
to the CLI explicitly and keeps its conversation memory within the group mount.

After interactive CLI model selection, the terminal asks for missing optimization
settings and saves each answer to that run before retrying preparation. Known
settings are reused within the run; supported instance presets and serving images
use the existing defaults. JSON commands return `needs_input` without prompting,
as before. Preparation displays the resulting configuration and explicit approval
command; it does not submit an optimization job.

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
explicit selection by model name or unambiguous shortlist number, persist it with `select_model`,
verify the saved run with `get_run_status`, and inspect `optimization_preparation`.
Selection automatically prepares optimization. Ask only for `needs_input` fields via
`prepare_optimization`, then summarize the saved configuration and ask for study approval.
`update_workload` can remember `gpu_type`, input/output tokens, concurrency, prompt count,
seed and frameworks alongside GPU costs. Run-specific preparation answers survive reconnects.
Python downloads the selected revision's card, researches heads with Tavily, asks Nemotron to
inspect sourced claims, and enables a head only when an official target launch agrees with
Nemotron's verified target/runtime pair. Concrete serving arguments preserve documented options.
Files live under `<state-root>/<run-id>/optimization-prepared/`: `intent.json`, `request.json`,
`research.json` and `selected-model-card.md`. The installed runtime needs llm-optimizer's remote
contracts to validate requests. Research/configuration failures preserve selection and block
submission; retry preparation after correcting them.
An absent optimizer must be reported as not connected. GPU work always requires explicit approval;
selection does not authorize a study.

## Security and limits

This is a single-user local prototype. The installed runtime is read-only and deliberately omits
`.env`, source checkout metadata, artifacts, and Docker access. The state mount is writable because
the existing CLI persists runs, selections, feedback, and revisions there. NanoClaw's default agent
container still includes its own container-local tools, so this is not a public multi-tenant design.

Live conversation validation requires Docker plus a provider credential. The Python MCP contract can
be validated without either using the test suite below.

Optional inference optimization setup and contracts are documented in [optimization.md](optimization.md).

Status responses are deliberately compact: raw model-card excerpts and large runtime
records can exceed OpenCode's tool-output window and hide the next conversation step.
Use `get_active_model_evidence` with an optional explicit `run_id` for detailed historical
or active evidence. Status retains source URLs, evidence classifications, method coverage,
memory assumptions and aggregate rejection reasons. The next selection question comes first.

The optional controller connection can be supplied through the operator-owned
`runtime/config/optimizer.json`; see [optimization.md](optimization.md). A verified SQLite
connection is not a verified worker image, GPU configuration, or live benchmark. Start with
dry-run defaults and do not start the external reconciliation loop as a connection test.

After selecting a model, inspect `optimization_preparation`: present saved paths and
summary when `prepared`; ask only for `needs_input` fields otherwise. Supply missing
infrastructure under `prepare_optimization.execution`. Explain `insufficient_evidence`
or `preparation_failed` instead of claiming the JSON is ready. Preparation preserves
the pinned card and cited research in the run directory and never submits work.

Discovery accepts model requirements alone, such as a 4B size and speculative-decoding
preference. GPU memory, price, budget, license and benchmark workload are optional.
Missing deployment information is reported as unknown rather than blocking candidates.
After selection, answer `optimization_preparation.needs_input` with `prepare_optimization`;
these are deployment questions, not recommendation clarification. Supplied hardware and
budget limits remain enforced. Infrastructure overrides are `optimizer_bucket`,
`optimizer_preset`, and `optimizer_images` (a JSON object encoded as a string).


vLLM, SGLang and TRT-LLM now have default optimizer worker images using the
`expanded-v1` tag. Known H100/H200/L40S platform and GPU count combinations
also have default presets (see optimization.md). Ask for image/preset overrides only
when needed; configured overrides take precedence over built-in defaults.

Optimization preparation now uses llm-optimizer's full expanded framework search profile.
Read `search_profile`, `search_regime`, `server_search_dimensions`, and `worker_requirements`
in the preparation summary. Concurrency 32 uses the batched search domains while the
client workload stays at 32. Updated worker images are required for config materialization
and compressed payloads; do not describe image capabilities as verified from their names.

Deployment and previous-build backup for the installed runtime update:
[nanoclaw-deployment-20261018.md](nanoclaw-deployment-20261018.md).

Optimization preparation accepts typed `dry_run` and `trials` controls. A request such as
"Prepare live optimization with two trials per framework" maps to
`prepare_optimization(run_id=..., dry_run=false, trials=2)`. Preparation preserves other
saved settings and requires separate approval before submission.

### Status and conversation cost

`get_optimization_status` accepts recommendation run IDs and linked optimizer study IDs.
By default it returns local controller state, provider IDs, errors and progress without S3
artifact reads. Use `include_results: true` for an explicit benchmark-results request.
Unexpected tool failures return an error without closing the MCP connection.

The installed OpenCode provider detects the `hf-model-watch` MCP server and applies a
32,768-token context window (or a smaller configured window), at most 4,096 output tokens,
automatic compaction with tool-output pruning, and four agent steps per turn. Shell recovery
and subagent delegation are disabled for this group. Other groups retain their prior policy.
The first turn after deployment can compact the existing large history; this is not a guarantee
of a 32K bound on that initial summarization call or a fixed chat response time.
Workload preferences, recommendation runs, and optimizer studies remain on disk.

The local NanoClaw source change and focused tests are preserved in
[the provider patch](patches/nanoclaw-hf-chat-cost.patch) for reproducibility. It applies to
this installed provider version; review it before applying to a newer NanoClaw checkout.
Deployment and validation details are in [the status/cost update](nanoclaw-status-cost-20261019.md).

### Automatic controller startup

The installed optimizer runs as the `hf-model-watch-optimizer.service` user service, with
three concurrent framework slots and the existing project `.env`. It starts with the user
service manager and restarts after failure. A companion `.path` service also starts it on
demand when an approved NanoClaw submission requests startup through the shared controller
directory. NanoClaw waits for a fresh heartbeat and database lease before queueing the study.
If it cannot establish readiness within 20 seconds, submission fails without creating a study.
Preparation and status checks do not start it. GPU launch still requires study approval.

Only one `serve` or `tick` may own the database: the controller now enforces a file lock
for the full reconciliation lifetime. Do not run a manual tick alongside the service.
The controller is a local CPU process; existing GPU worker images need no changes.
See [managed controller operations](managed-optimizer-controller.md) for status, logs and
installation/rollback details.

The HF context policy explicitly sets all three model limits: context 32768, input 28672,
and output 4096 (or smaller configured values). Setting only context is insufficient for
catalog models with a larger inherited input limit. Resumed sessions receive the same
limits for their saved model without changing the model selection.
