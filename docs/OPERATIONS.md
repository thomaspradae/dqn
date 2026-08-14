# Operations Manual

## Purpose

This manual defines supported procedures for operating, repairing, and extending
the research-compute system. It is organized by task and failure mode so future
operators and agents can act through the same paths instead of inventing local
workarounds.

Commands in this document should be run from the controller unless the procedure
explicitly says otherwise. The controller is currently `ofi1`
(`uace-ofi-01`). Verify live state before making changes; node and job state can
change faster than documentation.

## Access And Preconditions

Access is over Tailscale and SSH. The active controller user is `uace`. Do not
write passwords, Telegram tokens, or API keys into repository files, command
logs, documentation, or shell history.

Important deployed locations:

| Location | Purpose |
|---|---|
| `/home/uace/poormansops` | Cluster inventory, config generation, adoption, and control CLI source. |
| `/home/uace/.local/bin/poormans` | Controller wrapper for dashboard and control commands. |
| `/etc/poormans/nodes.env` | Authoritative node inventory. |
| `/etc/poormans/partitions.env` | Authoritative partition inventory. |
| `/etc/prometheus/prometheus.yml` | Prometheus service configuration. |
| `/etc/prometheus/file_sd/poormans_nodes.json` | Generated node scrape targets. |
| `/etc/slurm/slurm.conf` | Active Slurm configuration. |
| `/home/uace/dqn` | Deployed DQN project tree. |
| `/home/uace/researchops` | Deployed research status, run parsing, and cluster watcher tree. |
| `/home/uace/cluster/alerts` | Legacy DQN Telegram bot and alert scripts. |

Use short SSH timeouts for diagnostics against degraded nodes. A down worker
should produce a degraded report, not block an entire operator workflow.

## Routine Status Checks

Start with the high-level cluster view:

```bash
poormans status
poormans check
```

Use `poormans status` for a readable dashboard and `poormans check` for gate
failures. `check` failures should be classified by layer: SSH, command exec,
metrics, Slurm, or run-state.

Inspect Slurm directly when scheduling matters:

```bash
sinfo -Nel
squeue -o "%.18i %.12P %.32j %.12u %.10T %.10M %.6D %R"
scontrol show node <node>
scontrol show job <job_id>
```

Inspect Prometheus when machine metrics matter:

```bash
systemctl status prometheus
cat /etc/prometheus/file_sd/poormans_nodes.json
```

For experiment status, use the deployed research status commands where
available:

```bash
/home/uace/researchops/bin/dqn-status
/home/uace/researchops/bin/mlrun-status
```

If a status command hangs on a down node, diagnose it as a status timeout defect
and use narrower commands against reachable nodes.

## Node Lifecycle

### Bootstrap A Fresh Ubuntu Node

On a newly installed Ubuntu Server worker, run:

```bash
curl -fL https://thomaspradae.github.io/poormans/join | sudo bash
```

This command is served from the public `thomaspradae/poormans` repository. It
downloads the latest `poormans-node-bootstrap` Debian package from GitHub
Releases, installs `/usr/local/bin/poormans-join`, enables SSH, and prints
candidate controller-side adoption commands.

Expected output includes:

- hostname;
- candidate SSH users;
- IPv4 addresses;
- one or more `poormans adopt <user>@<ip>` commands;
- the reminder to run `poormans admit <alias>` after adoption.

This bootstrap is intentionally sealed. It does not write `/etc/poormans`, apply
Slurm, deploy Prometheus, copy secrets, join Tailscale, or admit the machine
into the cluster. Its job is only to make the new node reachable and easy to
adopt from a controller.

Test the public entrypoint without installing:

```bash
curl -fL https://thomaspradae.github.io/poormans/join | POORMANS_JOIN_DRY_RUN=1 bash
```

### Adopt A New Node

Use adoption for a fresh SSH-reachable node that should join inventory:

```bash
poormans adopt <ssh-target>
```

Expected behavior:

- SSH preflight runs;
- remote host, OS, CPU, RAM, disk, sudo, and Tailscale diagnostics are shown;
- the operator supplies alias, hostname, partition, role, Slurm name, CPU,
  memory, features, and weight;
- explicit confirmation is required;
- node-side join installs required services;
- inventory is updated;
- derived configuration is regenerated;
- Slurm dry-run output is shown.

State modified:

- `/etc/poormans/nodes.env`;
- generated poormansops artifacts;
- remote node packages and services.

Adoption does not silently apply Slurm. After adoption, run the admission and
validation steps deliberately.

For the common public-bootstrap path, the full operator sequence is:

```text
new node:
  curl -fL https://thomaspradae.github.io/poormans/join | sudo bash

controller:
  poormans adopt <user>@<ip>
  poormans admit <alias>
```

### Admit A Node Into Slurm

For a node already in inventory, use the guarded admission path:

```bash
poormans admit <node>
```

or perform the explicit sequence:

```bash
poormans deploy prometheus
poormans deploy slurm --dry-run
poormans deploy slurm --apply
poormans slurm-sync-configs
poormans slurm-onboard <node>
poormans slurm-auth-test <node>
poormans run-check <node>
poormans check
poormans status
```

`deploy slurm --apply` requires explicit confirmation. Review the dry-run before
applying. The apply path should back up the existing Slurm config and reconfigure
Slurm rather than restarting the whole cluster without cause.

Verification:

- SSH to the node works;
- `node_exporter` is active and scraped;
- Munge authentication succeeds;
- `slurmd` is active on the worker;
- `sinfo` shows the expected node and partition state;
- `poormans check` has no new failures.

### Inspect Worker Registry

The durable worker registry lives at:

```text
/etc/poormans/workers.json
```

Use it to inspect permanent and leased workers:

```bash
poormans workers
poormans worker show ofi1 --json
poormans worker show <worker-id> --json
```

The registry records stable worker identity, lifecycle, provider, state,
capabilities, Slurm node identity, and detach history. It deliberately separates
durable worker identity from active availability in `/etc/poormans/nodes.env`.

Expected lifecycle and state vocabulary:

```text
lifecycle: permanent | leased
provider:  owned | vast | runpod | friend | aws | <future-provider>
state:     attaching | ready | busy | offline | draining | detached | expired
```

Synchronize existing permanent workers only when the inventory or registry needs
reconciliation:

```bash
poormans workers sync --dry-run
poormans workers sync
```

Review dry-run output before persisting. This command can assign stable worker
IDs into active inventory.

### Validate Leased Worker Logic Without A GPU

GPU rental is not required to validate the control-plane shape. Use an existing
reachable CPU node for discovery-only validation:

```bash
poormans workers
poormans worker show old1 --json
poormans attach old1 --lease 6h --name lease-test-01 --provider friend --dry-run
poormans deploy slurm --dry-run
squeue -o '%i|%T|%M|%N|%R'
```

Expected result:

- `workers` lists permanent owned workers and their states;
- `worker show` returns a credential-free worker snapshot;
- `attach ... --dry-run` performs SSH/hardware discovery and prints a proposed
  leased worker;
- no inventory, registry, Slurm, Prometheus, or remote state is mutated;
- a CPU-only target must not accept `gpu` or `cuda` features if hardware
  discovery found no usable NVIDIA GPU.

This test validates the model, option parsing, discovery, and dry-run safety. It
does not validate GPU GRES registration or real leased-worker readiness.

### Attach A Leased Worker

Use `attach` for rented or borrowed machines that should temporarily join the
Slurm fabric.

Dry-run first:

```bash
poormans attach root@HOST \
  --port SSH_PORT \
  --name vast-h100-01 \
  --provider vast \
  --lease 6h \
  --dry-run
```

Inspect the proposed worker carefully:

- hostname and OS;
- CPU count and memory;
- GPU count, model, VRAM, driver, and CUDA;
- proposed Slurm name, partition, features, and weight;
- lease expiration;
- provider and lifecycle metadata.

Only after discovery is correct, attach for real:

```bash
poormans attach root@HOST \
  --port SSH_PORT \
  --name vast-h100-01 \
  --provider vast \
  --lease 6h
```

Attach is a temporary `adopt + admit`, not a new executor. It uses SSH/Tailscale
to bootstrap and configure the remote machine, then registers it as an ordinary
Slurm worker. After the worker is ready, experiments continue through the normal
run interface:

```bash
poormans run submit <contract.toml>
poormans runs
poormans run status <run-id>
poormans run cancel <run-id>
```

GPU contracts should request normal Slurm resources:

```toml
[resources]
gpus = 1
features = ["gpu", "cuda", "h100"]
```

### Detach A Leased Worker

Detach removes temporary availability while preserving history.

Always dry-run first:

```bash
poormans detach vast-h100-01 --dry-run
```

Then detach:

```bash
poormans detach vast-h100-01
```

If jobs are active, detach drains the Slurm node and exits without cancelling
work. Rerun detach after the jobs finish. The completion pass collects
artifacts, removes live Slurm/telemetry inventory, revokes the remote Munge key,
removes the live endpoint, and marks the durable worker record detached.

Detach does not wipe the remote machine, delete artifacts, power off the rental,
or erase historical worker/run provenance.

### Drain A Node

Drain before planned maintenance:

```bash
poormans drain-node <node>
```

Then verify:

```bash
sinfo -Nel
squeue -w <node>
```

Draining changes scheduler availability. It should not remove the node from
inventory and it should not stop Prometheus monitoring.

### Remove A Node

Generate and review the removal plan first:

```bash
poormans remove-plan <node>
```

Only remove the node after confirming that no active runs, queued evaluations,
or expected artifacts depend on it:

```bash
poormans remove-node <node>
```

After removal, regenerate and deploy derived configuration, then verify that
SSH aliases, Prometheus targets, and Slurm nodes no longer contain the removed
node.

### Restore A Node After Failure

Classify the failure before repairing:

1. Tailscale/SSH unreachable.
2. SSH reachable but commands fail.
3. Metrics missing but SSH works.
4. Munge fails.
5. `slurmd` fails.
6. Slurm controller marks node down despite healthy local services.

For the sixth case, verify the worker first, then inspect controller state and
logs before running any resume or reconfigure command. A node can be healthy at
the OS layer and still down in Slurm because of stale scheduler state.

## Slurm Operations

Use Slurm commands for scheduler truth.

Inspect nodes:

```bash
sinfo -Nel
scontrol show node <node>
```

Inspect jobs:

```bash
squeue
scontrol show job <job_id>
```

Cancel a job:

```bash
scancel <job_id>
```

Resume a drained or down node only after verifying SSH, Munge, `slurmd`, and
logs:

```bash
scontrol update NodeName=<node> State=RESUME
```

Synchronize configuration after inventory-driven Slurm changes:

```bash
poormans slurm-sync-configs
poormans slurm-auth-test <node>
```

Inspect logs:

```bash
journalctl -u slurmctld --since "1 hour ago"
journalctl -u slurmd --since "1 hour ago"
```

If jobs are pending with invalid account or partition messages, inspect the job
record and Slurm config rather than editing the experiment script first. The
scheduler owns those errors.

## Prometheus Operations

Prometheus target state is generated from node inventory. Do not hand-edit
`/etc/prometheus/file_sd/poormans_nodes.json` as a permanent fix.

Deploy updated Prometheus config:

```bash
poormans deploy prometheus
```

Verify service health:

```bash
systemctl status prometheus
cat /etc/prometheus/file_sd/poormans_nodes.json
```

If metrics are missing for one node:

1. Confirm SSH or Tailscale reachability.
2. Check `node_exporter.service` on that node.
3. Confirm port `9100` is reachable over Tailscale.
4. Confirm the node appears in generated file discovery.
5. Confirm Prometheus has reloaded or restarted successfully.

Prometheus unavailability affects dashboards and resource reports. It does not
by itself prove that an experiment failed.

## Experiment Operations

Current Atari runs are launched by Slurm scripts under `atari/*.sbatch`. A
typical serious run:

1. selects a partition, node, CPU and memory request;
2. changes into `/home/uace/dqn/atari`;
3. activates `/home/uace/dqn/.venv`;
4. calls `train_nature.py`;
5. writes to an explicit run directory;
6. records checkpoints and logs.

Until a single launcher exists, any new serious `*.sbatch` script must still
record enough information to satisfy [RUN_CONTRACT.md](RUN_CONTRACT.md).
At minimum, it must make the run id, output directory, log path, scheduler job,
configuration, and checkpoint policy discoverable.

Inspect a run by checking:

- Slurm job state;
- registry row;
- run metadata;
- recent log lines;
- metric files;
- checkpoint manifest;
- evaluation records.

When registry status and artifacts disagree, trust evidence in this order:

1. run-local terminal markers and final artifacts;
2. Slurm job state;
3. run-local metadata and logs;
4. evaluation ledgers and summaries;
5. registry status.

Then repair the registry to match the evidence. Do not leave completed runs
marked `running`.

## Evaluation Operations

Current DQN evaluation uses:

- `atari/eval_checkpoint.py`;
- `atari/auto_eval_milestones.py`;
- `/home/uace/dqn/atari/eval_queue.csv`;
- `/home/uace/dqn/atari/auto_eval_state.csv`;
- run-local `eval_results.csv`.

Evaluation queue states should distinguish queued, running, completed, failed,
and superseded requests. Repeated retries should have a visible reason. A stale
`running` evaluation should be marked failed or retried only after checking the
owner process, logs, and target checkpoint.

Final and best-checkpoint evaluations are durable run evidence. They should be
kept even when the training job is no longer active.

## Telegram Operations

The legacy DQN Telegram poller lives at:

```text
/home/uace/cluster/alerts/dqn_bot.py
```

Its cron entry polls commands every minute. It handles commands such as status,
resources, runs, archive, tails, best summaries, reports, evaluations, queue,
auto-eval, plots, history, and help. Modern DQN commands are delegated through
`research_telegram_command`.

The cluster node watcher lives at:

```text
/home/uace/researchops/cluster_watch/check_nodes.sh
```

It checks node reachability and sends Telegram only when node state changes.
Alert snapshots must be compact, plain text, and ANSI-free. Do not append raw
interactive `poormans` dashboard output to Telegram messages; that output
contains ASCII art and terminal color escape sequences. If a watcher needs a
cluster snapshot, call `poormans --telegram` after verifying that mode emits
plain text, or use watcher-owned summary lines.

This repository includes `scripts/fix_cluster_watch_snapshot.sh` as the targeted
operational patch for the deployed watcher bug where `status_snapshot()` called
the raw dashboard. Run it on the controller, then test:

```bash
bash -n /home/uace/researchops/cluster_watch/check_nodes.sh
```

`scripts/fix_poormans_telegram_mode.sh` is a secondary patch for the dashboard
itself if a future watcher or Telegram command intentionally wants to call
`poormans --telegram`.

Diagnose missing Telegram notifications by checking:

1. cron is installed and running;
2. the relevant script is executable;
3. private env configuration exists;
4. the script can reach Telegram;
5. the command source it reads does not block on a down node;
6. logs under the alert or watcher directory show the expected execution.

Never copy Telegram token values into repository docs or code. A legacy script
with hard-coded credentials should be migrated to private env configuration.

## Failure Recovery

### Node Unreachable

Check Tailscale status first, then SSH with a short timeout. If the node is
expected to be offline, drain it or keep it out of active scheduling. If it
should be online, repair host power, Tailscale, network, or SSH before touching
Slurm.

### Node Reachable But Metrics Down

Check `node_exporter.service`, port `9100`, firewall rules if any, and
Prometheus file discovery. Redeploy Prometheus targets from inventory after
inventory changes.

### Node Reachable But Slurm Down

Check Munge, `slurmd`, local Slurm config, controller logs, and `scontrol show
node`. If local services are healthy but the controller still marks the node
down, perform a controlled Slurm resume or reconfigure only after reading the
recorded reason.

### Leased Attach Fails Partway

Classify the phase before retrying:

- preflight/discovery failed: no inventory or registry mutation should have
  occurred; fix SSH, port, image, or credentials and rerun `attach --dry-run`;
- bootstrap failed before Tailscale IP discovery: inspect remote package/service
  logs, rerun `attach --dry-run`, then rerun attach after the remote issue is
  fixed;
- registry entry exists with `state=offline` or `attach-failed`: inspect
  `poormans worker show <name> --json` and the event history before retrying;
- inventory was written but Slurm readiness failed: run `poormans deploy slurm
  --dry-run`, `poormans slurm-auth-test <name>`, and `poormans run-check <name>`
  to identify whether the failure is config, Munge, `slurmd`, or data-plane
  initialization.

Do not hand-edit generated Slurm or Prometheus files to recover. Work from the
worker registry, active inventory, and the guarded `poormans` commands.

### Leased Worker Disappears

If a rental expires or disappears, Slurm owns the job/node failure signal.
`researchops` owns the run contract and checkpoint interpretation. Recovery is:

1. inspect Slurm node and job state;
2. inspect run state and latest durable checkpoint;
3. attach a replacement worker if needed;
4. resubmit/retry through `poormans run`, not through an SSH executor.

If the experiment is not checkpointable, treat the run as interrupted or failed.
Slurm cannot restore training state that the experiment never wrote.

### Leased Detach Fails Partway

Rerun `poormans detach <name> --dry-run` and inspect the worker record. If jobs
are still assigned, the correct state is draining and no jobs should be
cancelled by detach. If artifact collection failed, use `--skip-collect` only
after deciding that remote artifacts are already collected, disposable, or
recoverable by another route. If live inventory was removed but remote cleanup
failed, preserve the worker history and repair remote access separately.

### Job Missing But Registry Says Running

Inspect the run directory. Look for final artifacts, terminal log lines,
checkpoint manifests, evaluation completion markers, and failure traces. If the
run clearly finished, mark it completed. If it died without completion, mark it
failed or interrupted according to evidence. Do not rely on the stale registry
row.

### Metrics Stop Updating

Separate machine metrics from experiment metrics. If Prometheus metrics stop,
diagnose exporter and scrape health. If training metrics stop but machine
metrics continue, inspect the Slurm job and experiment logs.

### Evaluation Queue Stuck

Inspect the queued evaluation row, owner node, checkpoint path, evaluator logs,
and retry count. Supersede stale milestone requests when a newer request makes
them obsolete. Mark stuck `running` rows failed before retrying unless there is
clear evidence the evaluator is still active.

### Controller Restart

After controller restart, verify:

- Tailscale is up;
- SSH aliases still work;
- Prometheus is running and scraping;
- `slurmctld` is running;
- `slurmd` on the controller is running;
- `sinfo` and `squeue` return promptly;
- cron-driven Telegram and auto-eval jobs are active.

### Interrupted Resumable Run

Resume only if the run declares full or partial resume support and the required
checkpoint state exists. For a fully resumable run, verify model, optimizer,
counters, random state, data or replay state, and checkpoint manifest. Submit a
resume job through the launcher or documented Slurm path.

### Interrupted Non-Resumable Run

Do not pretend a model checkpoint is a full training checkpoint. For current
scratch Atari DQN, replay memory is not checkpointed, so exact resume is not
supported. Preserve the best/final artifacts, run evaluation if useful, and mark
the run completed, failed, or interrupted according to actual evidence.

## Maintenance

Routine maintenance should include:

- regenerate configs after inventory changes;
- run `poormans check`;
- inspect `poormans workers` for expired, offline, draining, or detached leased
  records;
- inspect Slurm down/drain reasons;
- reconcile stale registry rows;
- rotate or archive large logs;
- verify evaluation queue health;
- verify private env files exist for Telegram;
- remove hard-coded secrets from legacy scripts;
- test Telegram commands after status command changes;
- verify status commands time out on unreachable nodes.

Maintenance changes that modify behavior must update this manual or the system
architecture document in the same change.

## Extension Procedures

### Add A New Experiment Type

Define the adapter name, run directory layout, metadata writer, metrics writer
or adapter, checkpoint policy, evaluation policy, and registry fields. The new
experiment type is not integrated until it satisfies the minimum checklist in
[RUN_CONTRACT.md](RUN_CONTRACT.md).

### Add A Telegram Command

Identify the source of truth the command will read. Implement timeouts for any
remote calls. Keep formatting separate from state discovery. Document command
behavior and failure output in this manual when user-visible behavior changes.

### Add A Node Role Or Partition

Update `/etc/poormans/partitions.env` or inventory through the poormansops
workflow. Generate configuration, inspect dry-run output, apply through guarded
commands, sync Slurm worker config, and validate scheduling with a small test
job.

### Add A Provider Type

Provider is metadata, not a scheduler. Add a provider value such as `vast`,
`runpod`, `friend`, or `aws` only to describe lifecycle/provenance and support
operator filtering. Do not add provider-specific execution commands unless the
machine fundamentally cannot join Slurm and the architecture is deliberately
reopened.

### Add A Metric Backend

Do not replace the run contract with backend-specific state. Add an adapter that
exports or reads the common metrics event model. Prometheus remains machine
metrics unless a separate experiment-metrics pipeline is explicitly introduced
and documented.

## Command Reference

| Command | Owner | Side effect |
|---|---|---|
| `poormans status` | Dashboard/control wrapper | Reads cluster and service state. |
| `poormans check` | `poormansops` | Reads health gates; no intended mutation. |
| `poormans doctor <node>` | `poormansops` | Reads detailed node diagnostics. |
| `poormans adopt <ssh-target>` | `poormansops` | Adds node after guarded confirmation and remote join. |
| `poormans admit <node>` | `poormansops` | Runs guarded post-adoption admission sequence. |
| `poormans workers` | `poormansops` | Reads durable worker registry and Slurm-observed state. |
| `poormans worker show <node>` | `poormansops` | Reads one worker record or credential-free snapshot. |
| `poormans attach <ssh-target> --lease <duration>` | `poormansops` | Temporarily attaches a leased machine as a Slurm worker after confirmation. |
| `poormans detach <worker>` | `poormansops` | Drains and removes leased worker availability while retaining history. |
| `poormans deploy prometheus` | `poormansops` | Writes/reloads Prometheus config and file discovery. |
| `poormans deploy slurm --dry-run` | `poormansops` | Generates and diffs candidate Slurm config. |
| `poormans deploy slurm --apply` | `poormansops` | Backs up and applies Slurm config after confirmation. |
| `poormans slurm-sync-configs` | `poormansops` | Copies Slurm config to workers. |
| `poormans slurm-auth-test <node>` | `poormansops` | Validates Munge/Slurm auth path. |
| `poormans run-check <node>` | `poormansops` | Runs a node execution smoke check. |
| `sinfo` | Slurm | Reads scheduler node and partition state. |
| `squeue` | Slurm | Reads scheduler queue state. |
| `scontrol show job <job_id>` | Slurm | Reads detailed job state. |
| `scancel <job_id>` | Slurm | Cancels a job. |
| `journalctl -u slurmctld` | systemd/Slurm | Reads controller logs. |
| `journalctl -u slurmd` | systemd/Slurm | Reads worker logs. |

When in doubt, inspect first, identify the owner, then use the owner component's
procedure. Avoid one-off edits to derived files.
