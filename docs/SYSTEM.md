# System Architecture

## Purpose

The system exists to use several machines as one research-compute environment.
It should let an operator add machines, schedule work centrally, observe machine
health remotely, preserve experiment artifacts, evaluate checkpoints, and obtain
status or alerts without keeping a local laptop session alive.

The current system is real and deployed, but it grew in layers. The cleanup goal
is not to replace it with an imaginary platform. The goal is to define the
stable outer system clearly enough that experiment internals can change without
breaking operations, status reporting, or evaluation.

## Design Principles

Configuration should be inventory-driven. A node should be described once, then
that description should generate SSH configuration, Prometheus targets, Slurm
node records, and dashboard aliases.

Scheduling should be centralized. Slurm owns batch placement, job lifecycle, and
resource allocation. Experiment code may request resources through launch
configuration, but it should not implement its own cluster scheduler.

SSH and Tailscale are transport, bootstrap, inspection, and repair mechanisms.
They are not an execution backend. A worker is either made part of the Slurm
fabric or it is not used for normal experiment execution. This avoids rebuilding
submission, status, cancellation, stdout/stderr ownership, retries, resource
allocation, and job history outside Slurm.

Machine health and experiment health are separate signals. Prometheus answers
whether a machine is reachable, loaded, full, or hot. Run artifacts and
evaluation records answer whether an experiment is progressing or learning.

Runs should be reproducible and durable. A serious run needs identity,
configuration, metadata, logs, metrics, artifacts, checkpoints, and explicit
resume semantics. A run may use any ML framework internally, but the boundary it
exposes to the infrastructure must be standardized.

Remote reporting should read established state. Telegram commands and alerts
should summarize inventory, Slurm, Prometheus, run directories, registries, and
evaluation ledgers. They should not become independent sources of truth.

## Deployment Topology

The cluster currently uses Tailscale networking and SSH management. The
controller is `ofi1` (`uace-ofi-01`). The authoritative node inventory lives at
`/etc/poormans/nodes.env` on the controller.

Snapshot from the 2026-08-06 archaeology pass:

| Alias | Hostname | Tailscale IP | Role | Partition | Slurm name | CPU | Memory MB | Observed state |
|---|---|---:|---|---|---|---:|---:|---|
| `ofi1` | `uace-ofi-01` | `100.107.98.78` | controller | `office` | `ofi1` | 8 | 7700 | SSH up, metrics up, Slurm idle |
| `ofi2` | `uace-ofi-02` | `100.127.50.126` | worker | `office` | `ofi2` | 8 | 7700 | SSH down, metrics down, Slurm down |
| `old1` | `uace-old-01` | `100.80.3.43` | relay-worker | `relay` | `old1` | 4 | 3500 | SSH up, metrics up, Slurm down |
| `old2` | `uace-old-02` | `100.118.75.20` | worker | `relay` | `old2` | 4 | 3503 | SSH down, metrics down, Slurm down |

This table records an observed state, not a permanent truth. Operators must
verify live status with `poormans status`, Prometheus, and Slurm before taking
repair or scheduling actions.

## Component Architecture

### Repository Ownership

The system is split across three GitHub repositories with different trust
boundaries:

| Repository | Visibility | Owns | Does not own |
|---|---|---|---|
| `thomaspradae/poormans` | public | Public Ubuntu node bootstrap, GitHub Pages `join`, `poormans-node-bootstrap`, local SSH-prep helper. | Private inventory, Slurm, Prometheus, secrets, experiment execution. |
| `thomaspradae/poormansops` | private | Cluster control plane: inventory, adoption, admission, Slurm/Prometheus generation, worker registry, permanent and leased workers, attach/detach. | Experiment run contracts and learning-status semantics. |
| `thomaspradae/researchops` | private | Experiment run contract, Slurm-backed submission/status/cancel/collect, run manifests, worker provenance snapshots. | Node adoption, controller inventory, Slurm configuration, Prometheus targets. |

This DQN repository consumes those operational layers for Atari/DQN experiments.
It should not become the owner of public bootstrap, cluster admission, or generic
run-control machinery.

### Tailscale And SSH

Node communication is over Tailscale. `poormansops` generates SSH alias blocks
from the inventory so operators can address nodes by stable aliases such as
`ofi1`, `ofi2`, `old1`, and `old2`.

Generated SSH configuration exists on the controller at:

- `/home/uace/poormansops/generated/ssh_config_block`

SSH is used by adoption, diagnostics, Slurm worker onboarding, configuration
sync, project sync, status checks, and some experiment-status readers. Down
nodes must therefore be handled with short timeouts and clear degraded output.
A status command that blocks indefinitely on an unreachable node is a defect in
the status layer, not evidence that the cluster has no status model.

### `poormansops`

`poormansops` is the cluster-control layer. It owns inventory, node adoption,
configuration generation, Slurm deployment helpers, Prometheus discovery, worker
onboarding, and removal plans.

Its active deployed tree is:

- `/home/uace/poormansops`

Its primary state inputs are:

- `/etc/poormans/nodes.env`
- `/etc/poormans/partitions.env`

It generates or manages:

- SSH config blocks;
- Prometheus file discovery JSON;
- Prometheus configuration;
- Slurm node and partition fragments;
- sanitized dashboard alias maps;
- worker onboarding steps for Munge and Slurm.

Important controller commands include:

```text
poormans status
poormans generate
poormans check
poormans doctor <node>
poormans deploy [inventory|ssh|prometheus|slurm|all]
poormans adopt <ssh-target>
poormans admit <node>
poormans onboard
poormans slurm-onboard <node>
poormans slurm-sync-configs
poormans slurm-auth-test <node>
poormans sync-project <path>
poormans run-check <node>
poormans remove-plan <node>
poormans remove-node <node>
poormans drain-node <node>
poormans workers
poormans worker show <name-or-worker-id>
poormans attach <ssh-target> --lease <duration>
poormans detach <leased-worker>
```

`poormans adopt` is intentionally guarded. It preflights a remote node, gathers
diagnostics, prompts for node attributes, requires confirmation, runs a
node-side join, writes inventory, regenerates derived configs, and shows a Slurm
dry-run. It does not silently apply Slurm configuration. Applying Slurm is a
separate guarded operation.

### Public Node Bootstrap

Fresh Ubuntu workers can now be prepared through the public bootstrap repository:

- repository: `https://github.com/thomaspradae/poormans`;
- Pages entrypoint: `https://thomaspradae.github.io/poormans/join`;
- release package: `poormans-node-bootstrap`;
- installed helper: `/usr/local/bin/poormans-join`.

The public bootstrap command is:

```bash
curl -fL https://thomaspradae.github.io/poormans/join | sudo bash
```

This is not the private controller. It only installs the node bootstrap package,
enables SSH, prints hostname/users/IP addresses, and prints the controller-side
`poormans adopt <user>@<ip>` command. It does not write inventory, configure
Slurm, deploy Prometheus, copy secrets, join Tailscale, or admit the node into
the cluster.

The controller remains responsible for cluster admission. After public bootstrap
has made the new machine SSH-reachable, run `poormans adopt` and then
`poormans admit` from the controller.

### Worker Registry And Leased Workers

`poormansops` now has a durable private worker registry:

- `/etc/poormans/workers.json`

The registry assigns stable worker IDs and records lifecycle/provider/state
metadata independently from live Slurm availability. `/etc/poormans/nodes.env`
remains the active inventory used to generate SSH, Prometheus, and Slurm config.

Worker dimensions:

```text
lifecycle: permanent | leased
provider:  owned | vast | runpod | friend | aws | <future-provider>
state:     attaching | ready | busy | offline | draining | detached | expired
```

Permanent owned machines such as `ofi1` and `old1` are normal long-lived
workers. Leased or borrowed machines are temporary Slurm workers while attached.
The execution interface does not change between them:

```bash
poormans run submit experiment.toml
poormans runs
poormans run status <run-id>
poormans run cancel <run-id>
```

`poormans attach` is a temporary version of adoption plus admission. It SSHes to
a reachable machine, discovers hardware, bootstraps the node, establishes
Tailscale connectivity, registers the worker in inventory and registry,
generates telemetry and Slurm configuration, onboards Munge/Slurm, initializes
data-plane paths, and verifies Slurm readiness. `--dry-run` performs discovery
and prints the proposed leased worker without mutating local, remote, Slurm, or
telemetry state.

`poormans detach` removes temporary availability, not history. It drains first
and never cancels assigned jobs. When jobs are gone, it collects artifacts,
removes live inventory/telemetry/Slurm membership, revokes the remote Munge key,
removes the live endpoint from the registry, and retains durable worker/run
history.

### `poormans`

The name `poormans` currently refers to two related surfaces.

The first is a terminal dashboard. It prints cluster status, queries Prometheus
for CPU, memory, disk, uptime, load, and network counters, checks SSH
reachability, and uses Slurm commands when available to show jobs per node. This
surface is observational. It is not the scheduler and it is not the experiment
tracker.

The second is the controller wrapper at `/home/uace/.local/bin/poormans`, which
dispatches management commands into `~/poormansops/bin/poormansctl`. Commands
such as `deploy`, `adopt`, `slurm-onboard`, `remove-node`, or `drain-node`
modify cluster state and must be treated as operational changes.

Any future refactor should keep this distinction visible. Display-only status
commands and mutating control commands have different safety requirements.

### Prometheus And `node_exporter`

Prometheus runs on `ofi1` and scrapes `node_exporter` targets generated from the
poormans inventory.

Active paths:

- service: `prometheus.service`;
- config: `/etc/prometheus/prometheus.yml`;
- file discovery: `/etc/prometheus/file_sd/poormans_nodes.json`.

Node exporters are expected on each admitted machine. During the archaeology
pass, `node_exporter.service` was active on `ofi1` and `old1`; `ofi2` and
`old2` were unreachable.

Prometheus answers machine questions: reachability, scrape health, CPU load,
memory pressure, disk pressure, network activity, and similar resource signals.
It intentionally does not answer whether a DQN policy is improving, which
checkpoint is best, or whether an evaluation is complete. Those are experiment
questions and belong to the run/evaluation layer.

### Slurm

Slurm owns scheduling. The controller is `ofi1` and the cluster name is
`nucluster`.

Observed controller configuration:

- `slurmctld.service` running on `ofi1`;
- `slurmd.service` running on `ofi1`;
- config path `/etc/slurm/slurm.conf`;
- scheduler `sched/backfill`;
- select type `select/cons_tres` with `CR_Core`;
- accounting storage `accounting_storage/none`.

Partitions observed during archaeology:

| Partition | Nodes | Default | MaxTime |
|---|---|---|---|
| `office` | `ofi1,ofi2` | yes | infinite |
| `relay` | `old1,old2` | no | infinite |

Slurm configuration is generated from inventory and applied through guarded
`poormans` commands. `poormans deploy slurm --dry-run` should be used before an
apply. `poormans deploy slurm --apply` requires explicit confirmation and uses
`scontrol reconfigure` rather than a full service restart. Worker onboarding
copies shared Munge and Slurm state to workers.

Slurm state is operational state, not inventory. A node may be present in
inventory but marked `down` in Slurm. That condition should be repaired through
Slurm operations after verifying SSH, Munge, `slurmd`, and logs.

### Experiment Layer

The current experiment layer is concentrated around Atari DQN, but it already
contains more than one experiment format.

Important local files include:

- `atari/train_nature.py`;
- `atari/eval.py`;
- `atari/eval_checkpoint.py`;
- `atari/auto_eval_milestones.py`;
- `atari/research_status.py`;
- `atari/research_telegram_command.sh`;
- `atari/run_registry.csv`;
- `atari/*.sbatch`.

The deployed DQN registry is:

- `/home/uace/dqn/atari/run_registry.csv`

The current registry is useful for discovery, but it has been observed to be
stale. Some runs marked `running` have final artifacts and no corresponding
Slurm job. Therefore, registry rows should be reconciled against run-directory
evidence, Slurm state, and evaluation files before being used as authoritative
current activity.

`train_nature.py` already writes substantial artifacts: metadata, rewards,
training diagnostics, checkpoint files, checkpoint manifests, final model files,
and evaluation ledgers. The run contract in [RUN_CONTRACT.md](RUN_CONTRACT.md)
formalizes this outer interface so future experiment types can be added without
creating a new status universe.

### Evaluation Layer

The DQN evaluation layer evaluates checkpoints, records results, and can run
automatic milestone, final, and best-checkpoint evaluations.

Important files:

- `atari/eval_checkpoint.py`;
- `atari/auto_eval_milestones.py`;
- `atari/eval_queue.csv`;
- `atari/auto_eval_state.csv`;
- run-local `eval_results.csv`;
- auto-eval summaries under the deployed Atari tree.

Evaluation state is separate from training state. A training run can be
completed while some evaluations are queued, failed, superseded, or retried.
Status surfaces should make this distinction explicit.

### Telegram Layer

There are two practical Telegram systems.

The legacy DQN poller is deployed at:

- `/home/uace/cluster/alerts/dqn_bot.py`

It reads configuration from `/home/uace/cluster/alerts/config.env`, polls
Telegram updates, enforces chat identity, formats Telegram HTML responses, and
handles DQN/status commands. Modern DQN commands are delegated to
`~/.local/bin/research_telegram_command`, while older code paths still parse
Slurm and logs directly.

The node watcher is deployed at:

- `/home/uace/researchops/cluster_watch/check_nodes.sh`

It uses Tailscale and SSH checks to classify node availability, stores previous
state, and sends Telegram messages only when node state changes. This is machine
availability alerting, not experiment-learning status.

Telegram is a reporting and control surface. It must not become a primary store
for run state, node state, or evaluation state.

### `researchops`

`researchops` is a deployed observability and command-routing layer at:

- `/home/uace/researchops`

Important entrypoints include:

- `/home/uace/researchops/bin/dqn-status`;
- `/home/uace/researchops/bin/dqn-telegram-command`;
- `/home/uace/researchops/bin/mlrun-status`;
- `/home/uace/researchops/bin/dmref-status`;
- `/home/uace/researchops/bin/dqn-sync-state`.

`dqn-status` is effectively the deployed DQN status surface. `mlrun-status` is a
newer generic parser for env-file definitions under
`/home/uace/researchops/runs.d`. During archaeology, `mlrun-status` could hang
when a configured owner node was down. Status commands that SSH to remote owner
nodes should use fail-fast timeouts and report partial state.

## Sources Of Truth

| State | Authoritative source | Notes |
|---|---|---|
| Node inventory | `/etc/poormans/nodes.env` | Generate SSH, Prometheus, and Slurm node definitions from here. |
| Partition inventory | `/etc/poormans/partitions.env` | Generate Slurm partition definitions from here. |
| Generated SSH aliases | `/home/uace/poormansops/generated/ssh_config_block` | Derived from inventory. Do not edit as primary state. |
| Prometheus scrape targets | `/etc/prometheus/file_sd/poormans_nodes.json` | Derived from node inventory. |
| Slurm live scheduler state | Slurm controller on `ofi1` | Query with `sinfo`, `squeue`, and `scontrol`. |
| Durable worker identity | `/etc/poormans/workers.json` | Stable worker IDs, lifecycle, provider, state, capabilities, and detach history. |
| DQN run discovery | `/home/uace/dqn/atari/run_registry.csv` | Useful but currently stale; reconcile with run directories. |
| Run evidence | The run directory | Logs, metadata, metrics, checkpoints, and evals are durable evidence. |
| Evaluation queue | `/home/uace/dqn/atari/eval_queue.csv` | Queue state, not training state. |
| Auto-eval state | `/home/uace/dqn/atari/auto_eval_state.csv` | Tracks milestone/final/best eval processing. |
| Telegram bot configuration | Private env files on deployed host | Do not copy token values into code or docs. |

## System Flows

### Adding A Node

The simplest current fresh-node flow is:

```text
new Ubuntu worker:
  curl -fL https://thomaspradae.github.io/poormans/join | sudo bash

controller:
  poormans adopt <user>@<ip>
  poormans admit <alias>
```

The public bootstrap prepares the machine for SSH adoption and prints candidate
SSH targets. The operator then runs `poormans adopt <ssh-target>` from the
controller, records alias and Slurm attributes, confirms the adoption, and lets
the controller-side adoption flow copy and run its node-side setup as needed.
The controller then regenerates derived configuration.

Prometheus and Slurm admission are separate follow-up steps. Prometheus target
generation can be deployed independently. Slurm admission should be reviewed via
dry-run, applied through the guarded Slurm deploy path, synchronized to workers,
and validated with Munge and `slurmd` checks.

### Attaching A Leased Worker

A leased worker starts as a rented or borrowed machine that is reachable through
SSH. It must become an ordinary Slurm worker before normal experiments use it.

```text
leased machine:
  SSH reachable at root@HOST or user@HOST

controller:
  poormans attach root@HOST --lease 6h --name vast-h100-01 --provider vast --dry-run
  poormans attach root@HOST --lease 6h --name vast-h100-01 --provider vast
```

From the experiment layer's perspective, there is no `ssh` executor. The worker
is either a ready Slurm worker or unavailable. If a rental disappears, Slurm
reports node/job failure, `researchops` reads the run contract and latest
durable checkpoint, and the operator can attach another worker and resubmit or
retry through the normal Slurm-backed run interface. Exact training resume still
depends on experiment checkpoint semantics; Slurm cannot create missing replay
buffers or framework state.

### Submitting An Experiment

Today, serious Atari runs are submitted through individual `*.sbatch` scripts
that activate the deployed virtualenv and call `train_nature.py` with explicit
run directory and checkpoint settings. This is close to a launcher, but it is
not yet one canonical `submit_run --config ...` path.

The target flow is config to launcher to Slurm job to run directory. Slurm owns
placement and job id. The launcher owns run identity, resolved configuration,
initial metadata, and output paths. Experiment code owns metrics, artifacts, and
checkpoint writes.

### Monitoring An Experiment

Machine monitoring starts with Prometheus and Slurm. Experiment monitoring
starts with the run directory, registry index, and evaluation files. Telegram
and status commands should combine those sources without treating any display
surface as authoritative.

If a node is down, monitoring should degrade. It is acceptable to report that
owner-node evidence could not be fetched. It is not acceptable for a cluster-wide
status command to hang indefinitely because one owner node is unreachable.

### Evaluating A Checkpoint

Checkpoint evaluation begins with a checkpoint selector such as latest, final,
best candidate, or an explicit checkpoint path. The evaluator resolves the run,
executes the evaluation with a defined episode and seed policy, appends results,
updates queue state, and may produce a Telegram summary.

Evaluation results attach to the run but are not the same as training metrics.
They should remain queryable after the training job has finished.

### Recovering From Failure

Failures must be classified before repair:

- node failure: SSH, Tailscale, host power, exporter, Munge, or `slurmd`;
- scheduler failure: Slurm node state, queue state, invalid account, partition,
  controller, worker daemon, or Munge mismatch;
- process failure: job ended or crashed while node and scheduler are healthy;
- interrupted resumable run: all required training state exists and the run
  declares full or partial resume support;
- interrupted non-resumable run: final or best artifacts may be usable, but
  exact continuation is not supported.

Atari DQN is currently non-resumable because replay memory is not checkpointed.
Slurm requeue cannot fix that by itself.

## Current Limitations

The following limitations are known and should be treated as cleanup targets:

- duplicate research status implementations exist locally and in deployed
  `researchops`;
- the DQN registry contains stale `running` rows;
- some status paths can block on SSH to down nodes;
- scratch Atari DQN checkpointing does not save replay memory and therefore
  does not support exact resume;
- Telegram responsibilities are mixed between legacy parsing, modern delegated
  commands, node watching, and evaluation notifications;
- `ofi2` and `old2` were unreachable during archaeology, while `old1` was
  reachable but still marked down by Slurm;
- there is no single blessed experiment launcher;
- there is no one metrics JSONL format across experiment frameworks;
- a legacy alert script contained hard-coded Telegram credentials and should be
  cleaned up without copying the secret.

## Target Architecture

The intended steady state is:

- one inventory-driven node model;
- one scheduler path through Slurm;
- one launcher that creates run identity and submits jobs;
- one framework-independent run contract;
- one authoritative run registry/index with adapters for legacy records;
- one evaluation model that attaches results to runs;
- Telegram commands that read standardized state;
- timeout-safe status commands;
- clear separation between infrastructure code and experiment internals.

This target keeps the current deployed pieces. It standardizes the boundaries
between them so new experiment types can be added without rebuilding cluster
operations each time.
