# Managed optimizer controller

NanoClaw validates the selected request and approval, ensures the controller is healthy,
then submits the study. The existing llm-optimizer controller owns all provider execution,
Optuna work scheduling and result collection. Preparation and status remain read-only with
respect to service activation; neither authorizes GPU work.

The host runs `hf-model-watch-optimizer.service` with three framework slots. Its launcher
loads the existing `/mnt/c/Users/maorb/work/hf_validator/.env` and uses the existing
`/mnt/c/Users/maorb/work/bento_extend/llm-optimizer/.local/optimizer-controller.sqlite`.
The service is enabled for user-manager startup and restarts on failure. It requires the
WSL/Linux host and its user service manager to be running.

Because NanoClaw runs in a container, it requests startup through the already shared
database directory, without systemd access or host credentials in the agent:

1. An operator-owned `.controller-managed.json` sidecar enables managed startup.
2. On approved submission, the adapter accepts a fresh healthy heartbeat plus a held
   database lock; otherwise it touches `.controller-start`.
3. `hf-model-watch-optimizer.path` activates the service. The controller publishes an
   atomic `.controller-health.json` heartbeat every two seconds and clears the signal.
4. Submission waits up to 20 seconds for readiness and the configured three-slot
   concurrency. It creates no study if readiness fails.

The heartbeat verifies the local process and lease, not provider credentials or GPU quota.
A process lock on `.controller.lock` covers all of `serve`, including waits between ticks,
and individual manual `tick` calls. A second controller is rejected before provider calls.
Submit/status/cancel operations can still access the SQLite queue while reconciliation runs.
An unmanaged installation preserves the original CLI behavior.

Check the service and logs:

```bash
systemctl --user status hf-model-watch-optimizer.service
journalctl --user -u hf-model-watch-optimizer.service -n 50 --no-pager
```

After changing `.env`, restart the service so it loads the new values:

```bash
systemctl --user restart hf-model-watch-optimizer.service
```

Do not start a second `serve` or `tick` terminal. Stopping the local controller does not cancel
already-running Nebius jobs; use the existing study cancellation operation for that.

The implementation lives in llm-optimizer `remote/service.py`, `Controller.exclusive/tick`,
and its existing remote CLI. hf-model-watch gates approved submissions in
`optimization/workflow.py`. The cross-repository source patch is preserved in
[optimizer-managed-controller.patch](patches/optimizer-managed-controller.patch).
The installed units are under `/home/maorb/.config/systemd/user/`; the launcher is
`/home/maorb/hf-model-watch-nanoclaw/controller/start.sh`. No serving image rebuild is needed.

Deployment backups, including the controller database, previous modules and group instructions:
`/home/maorb/hf-model-watch-nanoclaw/backups/before-managed-controller-20261019T103009Z/`.
For rollback, stop and disable both managed units before restoring previous modules and
removing the managed registration. Do not restore an older database over newer studies.

Validation covered 152 focused Python tests, including managed startup, stale heartbeat,
startup failure blocking submission, malformed registration, duplicate-process exclusion
and existing remote, spot pricing, expanded search, preparation and conversation flows. Ruff checks passed for changed optimizer
modules and new controller tests. Activation from the agent container made the real host
service healthy in 637 ms against a verified empty queue, with no new runs or jobs. A manual
tick was then rejected while the service held the lease. No GPU jobs were launched by these
checks. GPU/provider behavior and recovery from an actual host reboot were not tested.
