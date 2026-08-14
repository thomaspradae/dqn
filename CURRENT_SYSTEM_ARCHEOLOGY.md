# Current System Archaeology

Date: 2026-08-06

Canonical docs created from this evidence:

- [AGENTS.md](AGENTS.md)
- [docs/SYSTEM.md](docs/SYSTEM.md)
- [docs/RUN_CONTRACT.md](docs/RUN_CONTRACT.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)

This file is a dated archaeology snapshot. Use it as evidence for what was
observed on 2026-08-06, not as the primary interface or operating manual.

Scope: local DQN repo, live controller `uace-ofi-01`, reachable worker
`uace-old-01`, Slurm, poormans/poormansops, Prometheus/node_exporter, SSH,
Telegram bots, and current experiment/run tracking.

No repair actions were taken. This is an inventory and behavior report.

## Executive Summary

The current system is real but split across several layers:

```text
Tailscale + SSH
  -> poormansops inventory/adoption
  -> Prometheus + node_exporter machine metrics
  -> Slurm scheduling
  -> DQN/researchops run registries and eval queues
  -> Telegram poller / node watcher / auto-eval notifications
```

The strongest parts are:

- `poormansops` has a usable inventory-driven cluster model.
- `poormans` gives a terminal cluster dashboard and controller commands.
- Prometheus on `ofi1` scrapes node exporters via generated file discovery.
- Slurm is installed and active on `ofi1`; the cluster config is inventory-driven.
- The DQN Atari runs produce useful artifacts: metadata, rewards, diagnostics,
  checkpoints, eval results, and final auto-eval summaries.
- Telegram integration exists and can route status/eval/queue commands.

The weakest parts are:

- Current node health is degraded: only `ofi1` and `old1` are reachable; Slurm
  marks `ofi2`, `old1`, and `old2` down.
- The DQN `run_registry.csv` is stale: several runs marked `running` are actually
  finished or no longer in Slurm.
- There are two overlapping research status systems: repo-local
  `atari/research_status.py` and deployed `/home/uace/researchops`.
- `research_status status all` can hang for minutes when nodes in the registry
  are down.
- Atari resume is intentionally disabled in `train_nature.py` because the replay
  buffer is not checkpointed. Slurm can requeue a job, but the current trainer
  cannot exactly resume the training state.
- Telegram has both a legacy bot and newer research command routing. It works,
  but responsibilities are mixed.
- One legacy alert script contains hard-coded Telegram credentials. That should
  be treated as a security cleanup item.

The system is already close to the proposed future shape, but not yet cleanly
standardized. The right next move is not to build a new platform. It is to make
one run contract and one launcher that reuses these pieces.

## Machine Inventory

Source of truth: `/etc/poormans/nodes.env` on `ofi1`.

| Alias | Hostname | Tailscale IP | Role | Partition | Slurm name | CPU | Memory MB | Live status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `ofi1` | `uace-ofi-01` | `100.107.98.78` | controller | `office` | `ofi1` | 8 | 7700 | SSH up, metrics up, Slurm idle |
| `ofi2` | `uace-ofi-02` | `100.127.50.126` | worker | `office` | `ofi2` | 8 | 7700 | SSH down, metrics down, Slurm down |
| `old1` | `uace-old-01` | `100.80.3.43` | relay-worker | `relay` | `old1` | 4 | 3500 | SSH up, metrics up, Slurm down |
| `old2` | `uace-old-02` | `100.118.75.20` | worker | `relay` | `old2` | 4 | 3503 | SSH down, metrics down, Slurm down |

Live `poormans status` on `ofi1` reported:

```text
ofi1   up     up     up       idle           low      low      controller/office/none
ofi2   down   down   down     down*/warn     N/A      N/A      worker/office/Node unexpectedly rebooted
old1   up     up     up       down/warn      low      low      relay-worker/relay/Not responding : Not responding
old2   down   down   down     down*/warn     N/A      N/A      worker/relay/Not responding : Not responding
```

Live `poormans check` failed with six problems, all from `ofi2` and `old2`
failing SSH, exec, and metrics gates.

## Network And SSH

SSH is over Tailscale. `poormansops` generates this SSH block:

```sshconfig
Host ofi1 uace-ofi-01
  HostName 100.107.98.78
  User uace
  StrictHostKeyChecking accept-new

Host ofi2 uace-ofi-02
  HostName 100.127.50.126
  User uace
  StrictHostKeyChecking accept-new

Host old1 uace-old-01
  HostName 100.80.3.43
  User uace
  StrictHostKeyChecking accept-new

Host old2 uace-old-02
  HostName 100.118.75.20
  User uace
  StrictHostKeyChecking accept-new
```

The controller-side generated block exists at:

- `/home/uace/poormansops/generated/ssh_config_block`

The local controller wrapper `~/.local/bin/poormans` dispatches management
commands to `~/poormansops/bin/poormansctl`; unknown/no-arg dashboard calls fall
back to `/usr/local/bin/poormans`.

Current SSH reality:

- `ssh uace@100.107.98.78` works.
- `ssh uace@100.80.3.43` works.
- `ofi2` and `old2` are not reachable right now.
- Tailscale on `ofi1` reports `ofi2` last seen 5 days ago and `old2` last seen
  9 days ago.

## What `poormans` Does

There are two related meanings of `poormans`.

### 1. The Pretty Dashboard

Installed locally at `/home/t/.local/bin/poormans` and on `ofi1` through
`/usr/local/bin/poormans` / `~/.local/bin/poormans`.

It:

- Prints an ASCII terminal dashboard.
- Chooses a healthy Prometheus endpoint from configured candidates.
- Queries Prometheus for CPU, RAM, disk, uptime, load, and network throughput.
- Checks SSH port reachability using TCP connect.
- Uses `squeue` when available to show Slurm jobs per node.
- Knows the four node labels and Tailscale addresses.
- Supports `--version` and `--telegram`.

Limits:

- It is a status dashboard, not the scheduler.
- It tells whether SSH/metrics/Slurm look up, not why an experiment is learning.
- Its node list can be overridden by `/etc/poormans/nodes.env`, but older copies
  still contain hard-coded fallback node defaults.
- It can wait on down nodes; the full scrape took about 35 seconds live.

### 2. The `poormansops` Control CLI

Controller wrapper: `/home/uace/.local/bin/poormans`

It dispatches commands such as:

```text
status
generate
check
doctor <node>
deploy [inventory|ssh|prometheus|slurm|all]
adopt <ssh-target>
onboard
slurm-onboard <node>
slurm-sync-configs
slurm-auth-test <node>
sync-project <path>
run-check <node>
remove-plan <node>
remove-node <node>
drain-node <node>
```

It owns the cluster inventory and generates:

- SSH config blocks.
- Prometheus file discovery JSON.
- Prometheus config.
- Slurm node/partition fragments.
- Public alias maps for sanitized dashboards.

Current source tree:

- `/home/uace/poormansops`
- Private repo according to its README.

Current active state lives under:

- `/etc/poormans/nodes.env`
- `/etc/poormans/partitions.env`
- `/etc/prometheus/file_sd/poormans_nodes.json`
- `/etc/slurm/slurm.conf`

## What `poormans adopt` Does

`poormans adopt <ssh-target>` is the interactive fresh-node admission helper.

It does:

1. SSH preflight to the target.
2. Collects remote diagnostics:
   - hostname
   - OS
   - kernel
   - architecture
   - CPU count
   - physical RAM
   - suggested Slurm RAM
   - disk free
   - sudo availability
   - Tailscale status/IP
3. Prompts for:
   - alias
   - stored hostname
   - Slurm partition
   - role
   - Slurm node name
   - CPUs
   - memory
   - features
   - Slurm weight
4. Requires explicit confirmation: `ADOPT-<alias>`.
5. Copies `poormans-join` to the node.
6. Runs node-side join with `sudo`.
7. Discovers the Tailscale IPv4 address.
8. Writes the node into the inventory.
9. Regenerates configs.
10. Shows a Slurm dry-run.
11. Prints the next guarded steps.

It does not silently apply Slurm. After adoption it tells you to run, as
appropriate:

```text
poormans deploy prometheus
poormans deploy slurm --apply
poormans slurm-sync-configs
poormans slurm-onboard <alias>
poormans slurm-auth-test <alias>
poormans run-check <alias>
poormans check
poormans status
```

There is also `poormans admit <node>`, implemented as a post-adoption helper:
it dry-runs Slurm, deploys Prometheus, applies Slurm after guarded confirmation,
syncs configs, onboards the node, auth-tests, run-checks, and prints status.

### Node-Side Join

`poormans join` / `poormans-join` is the node-side installer. It is designed to
be idempotent.

It installs/enables:

- `openssh-server`
- Tailscale if missing
- `prometheus-node-exporter`
- `munge` and `slurmd` by default

It keeps `slurmd` stopped until the controller has pushed shared config and
Munge state. Use `--no-slurm` for SSH/metrics-only nodes.

## Prometheus And Metrics

Prometheus runs on `ofi1`:

- Service: `prometheus.service`
- Config: `/etc/prometheus/prometheus.yml`
- Node discovery: `/etc/prometheus/file_sd/poormans_nodes.json`
- Scrape interval: 5 seconds for `prometheus` and `node` jobs.

Node exporter:

- Service: `node_exporter.service` on `ofi1`.
- Service: `node_exporter.service` on `old1`.
- Targets are generated from `/etc/poormans/nodes.env`.

Prometheus current target intent:

```json
[
  {"targets": ["100.107.98.78:9100"], "labels": {"nodename": "ofi1"}},
  {"targets": ["100.127.50.126:9100"], "labels": {"nodename": "ofi2"}},
  {"targets": ["100.80.3.43:9100"], "labels": {"nodename": "old1"}},
  {"targets": ["100.118.75.20:9100"], "labels": {"nodename": "old2"}}
]
```

Prometheus answers machine-health questions:

- Is the node exporting?
- CPU/RAM/load/disk/network?
- Is a node hot enough to avoid eval?

It does not answer experiment-learning questions. That is handled by
`research_status.py` reading run artifacts.

## Slurm

Slurm controller:

- Host: `ofi1` / `uace-ofi-01`
- `slurmctld.service`: running
- `slurmd.service`: running on `ofi1`
- Config: `/etc/slurm/slurm.conf`
- Cluster name: `nucluster`
- Scheduler: `sched/backfill`
- Select type: `select/cons_tres`, `CR_Core`
- Accounting: `accounting_storage/none`

Partitions:

| Partition | Nodes | Default | MaxTime |
| --- | --- | --- | --- |
| `office` | `ofi1,ofi2` | yes | infinite |
| `relay` | `old1,old2` | no | infinite |

Current live Slurm state:

```text
ofi1  idle
ofi2  down*      Node unexpectedly rebooted
old1  down       Not responding : Not responding
old2  down*      Not responding : Not responding
```

Current live Slurm queue:

```text
56|office|dmref_bko_ofi2|uace|PENDING|0:00|1|(None)
```

Detailed `scontrol show job 56` says:

- Job name: `dmref_bko_ofi2`
- Requested node: `ofi2`
- Command: `/home/uace/dqn/atari/jobs/dmref_breakout_ofi2_simple.sbatch`
- WorkDir: `/home/uace/dqn/reference/DeepMind-Atari-Deep-Q-Learner`
- Restarts: 2
- Submit time: 2026-07-19 20:50 UTC
- It is waiting on an unhealthy target node and controller logs repeatedly
  report `JobId=56 has invalid account`.

Important Slurm design details:

- `poormans deploy slurm --dry-run` writes and diffs a candidate config.
- `poormans deploy slurm --apply` requires typing `APPLY-SLURM`.
- The apply path backs up `/etc/slurm/slurm.conf`.
- It uses `sudo scontrol reconfigure`, not a full service restart.
- Worker onboarding copies the shared Munge key and Slurm config to workers.
- `slurm-sync-configs` copies live Slurm config to worker nodes.

Current `old1` finding:

- SSH works.
- `node_exporter` works.
- `munge` is active and local `munge -n | unmunge` succeeds.
- `slurmd` is active and listening on port 6818.
- Slurm controller still marks `old1` down with reason from 2026-06-30.
- This may be a stale Slurm node state needing a controlled resume/reconfigure,
  but this report did not mutate state.

## Job Execution Today

There are several job execution styles in the history:

- CartPole Slurm job: `cartpole/run.slurm`.
- Atari Slurm jobs: `atari/*.sbatch`.
- External SB3 run on `old1`.
- DeepMind reference run under `/home/uace/researchops/runs.d`.
- Older/nohup/SB3 paths in archived registry rows.

The main DQN Slurm scripts generally do:

```text
#SBATCH job/partition/node/resources/log paths
cd /home/uace/dqn/atari
source /home/uace/dqn/.venv/bin/activate
export OMP_NUM_THREADS=...
python train_nature.py ... --run-dir ... --checkpoint-every ...
```

This is close to a standard launcher, but it is still script-by-script. There is
not yet one blessed `submit_run --config ...` entrypoint.

## DQN Run Tracking

The canonical DQN registry is:

- `/home/uace/dqn/atari/run_registry.csv`

Local repo copy:

- `atari/run_registry.csv`

Fields include:

- `run_id`
- `algo`
- `game`
- `env_id`
- `owner_node`
- `ssh_target`
- `type`
- `job_id`
- `run_dir`
- `log`
- `seed`
- `status`
- `target_steps`
- `step_unit`
- `checkpoint_every`
- `eval_every_steps`
- `project_dir`
- `venv_path`
- `eval_script`
- `checkpoint_pattern`
- `metric_type`
- `notes`

Supported run protocols:

- `scratch_dqn`: reads `rewards.csv`, `checkpoints.csv`, `eval_results.csv`,
  `q_net_ep*.pt`, `q_net_final.pt`.
- `sb3`: reads `monitor.csv`, `*.zip`, and SB3 log text.

Current registry staleness:

- `breakout_v3_seed20260621_rewardclip_dm50mdec` is marked `running`, but its
  Slurm job is not in the queue and its run folder has `AUTOEVAL_DONE`,
  `q_net_final.pt`, `checkpoint_final.pt`, and final log line `SBATCH END
  2026-06-27T23:30:11+00:00`.
- `breakout_sb3_12_5m_seed20260609` is marked `running`, but the run has
  `final_model.zip`, reached 12.5M timesteps, and ended on
  2026-06-12T15:08:22+00:00.
- `breakout_v4_seed20260621_dmrmsprop_bilinear` is marked `running`, but its
  owner `ofi2` is currently unreachable. Its final Telegram summary exists
  under `/home/uace/dqn/atari/auto_eval_summaries`, suggesting the registry may
  also be stale there.

## Training Artifacts

`train_nature.py` writes:

- `RUN_METADATA.json`
- `run_metadata.json`
- `rewards.csv`
- `training_diagnostics.csv`
- `q_net_ep<episode>.pt`
- `checkpoint_ep<episode>.pt`
- `checkpoints.csv`
- `q_net_final.pt`
- `checkpoint_final.pt`

Metadata includes:

- command and argv
- args
- seed
- env id
- budget unit and budget
- optimizer metadata
- resize interpolation
- git commit/dirty state
- SHA-256 of key code files
- Python and package versions
- CUDA/device info
- explicit flags:
  - `replay_buffer_checkpointed: false`
  - `resume_supported: false`

For the completed `breakout_v3_seed20260621_rewardclip_dm50mdec` run:

- Created: 2026-06-21T23:02:11Z
- Final log: 2026-06-27T23:30:11Z
- Final agent step: 34,820,550
- Final raw env step: 139,587,075
- Final episode: 99,999
- Final checkpoint exists.
- `AUTOEVAL_DONE` exists.
- Final 30-episode eval mean: 1.40.
- Best selected 30-episode checkpoint eval: `ep6200`, mean 49.00.

## Resume Reality

`train_nature.py` accepts legacy-looking resume arguments, but then rejects them:

```text
--resume is disabled because replay memory is not checkpointed
--start-* counters are disabled because replay memory is not checkpointed
```

This is important:

- Slurm can requeue jobs.
- The training loop writes model and optimizer checkpoints.
- The replay buffer is not saved.
- Therefore exact Atari training resume is not currently supported.

The run contract should say this plainly. Either future jobs must checkpoint
replay memory and signal-handled shutdowns, or they should be treated as
non-resumable experiments whose final/best artifacts are the only durable output.

## Evaluation And Auto-Eval

DQN eval components:

- `atari/eval.py`: runs a model for N episodes and prints summary stats.
- `atari/eval_checkpoint.py`: resolves registry runs and checkpoint selectors,
  runs low-priority eval on the owner node, appends `eval_results.csv`, and
  manages `eval_queue.csv`.
- `atari/auto_eval_milestones.py`: every milestone, queues latest-checkpoint
  evals; for completed runs, does final and best-checkpoint 30-episode evals,
  writes summaries, and sends Telegram notifications.
- `atari/telegram_notify_report.py`: sends a summary through Telegram using
  the existing bot config.

Current cron on `ofi1`:

```cron
*/15 * * * * /home/uace/.local/bin/auto_eval_milestones >> /home/uace/dqn/atari/logs/auto_eval_milestones.log 2>&1
```

Queue file:

- `/home/uace/dqn/atari/eval_queue.csv`

State file:

- `/home/uace/dqn/atari/auto_eval_state.csv`

Auto-eval has worked historically, but the queue also shows past fragility:

- Some final milestone requests retried hundreds of times before being marked
  superseded.
- Some evals failed because deployed `eval.py` did not yet accept
  `--resize-interpolation`.
- Stale `running` queue entries had to be marked failed.

## Telegram Bot Layer

There are two practical Telegram systems.

### 1. Legacy DQN Telegram Poller

Path:

- `/home/uace/cluster/alerts/dqn_bot.py`

Config:

- `/home/uace/cluster/alerts/config.env`
- Contains `BOT_TOKEN`, `CHAT_ID`, and `ATARI_DIR`.

Cron:

```cron
* * * * * /home/uace/cluster/alerts/dqn_bot.py poll
*/5 * * * * /home/uace/cluster/alerts/dqn_watch_smart.sh
```

What it does:

- Polls Telegram `getUpdates`.
- Enforces chat id.
- Formats responses as Telegram HTML.
- Handles `/status`, `/poormans`, `/resources`, `/runs`, `/archive`, `/tail`,
  `/best`, `/sb3`, `/report`, `/eval`, `/queue`, `/auto-eval`, `/plot`,
  `/history`, and `/help`.
- Delegates modern DQN commands to `~/.local/bin/research_telegram_command`.
- Falls back to legacy Slurm/log parsing for active jobs, history, and plots.
- Sends job-change/hourly/milestone alerts via `dqn_watch_smart.sh`.

Limits:

- It only knows `ofi1` and `ofi2` in its internal legacy node map.
- It mixes old log scraping with new research-status delegation.
- Long commands can block for a long time if remote nodes are down.

Security note:

- `/home/uace/cluster/alerts/dqn_alert.sh` contains hard-coded Telegram
  credentials. That should be removed or replaced with `config.env`.

### 2. Cluster Node Watcher

Path:

- `/home/uace/researchops/cluster_watch/check_nodes.sh`

Config:

- `/home/uace/researchops/cluster_watch/nodes.tsv`
- `/home/uace/researchops/cluster_watch/telegram.env`

Cron:

```cron
* * * * * /home/uace/researchops/cluster_watch/check_nodes.sh >> /home/uace/researchops/cluster_watch/cron.log 2>&1
```

What it does:

- Uses `tailscale ping` and SSH to classify nodes as up/down.
- Stores previous state in `state.tsv`.
- Sends Telegram only when node state changes.
- Includes a short status snapshot from `poormans` or `research_status`.

This is machine availability alerting, not experiment learning status.

## Researchops

Path:

- `/home/uace/researchops`

Purpose according to README:

- private experiment observability
- DQN research status reporting
- Telegram command/status glue
- cluster-wide lightweight run-state sync
- eval queue/status reporting

Important entrypoints:

- `/home/uace/researchops/bin/dqn-status`
- `/home/uace/researchops/bin/dqn-telegram-command`
- `/home/uace/researchops/bin/mlrun-status`
- `/home/uace/researchops/bin/dmref-status`
- `/home/uace/researchops/bin/dqn-sync-state`

`dqn-status` is effectively the deployed version of the DQN status script.

`mlrun-status` is a newer generic run status parser for env-file definitions in
`/home/uace/researchops/runs.d`. Current configured run:

- `dmref_bko_ofi2`
- kind `deepmind_torch7`
- node `ofi2`
- job name `dmref_bko_ofi2`
- target 50M steps
- checkpoint path under the DeepMind reference checkout

Limit:

- `mlrun-status` hung while trying to SSH to `ofi2`, which is currently down.
  It needs timeouts/fail-fast behavior.

## Current Experiment Reality

Live scheduler:

- One pending job: `dmref_bko_ofi2` on requested node `ofi2`.
- No active DQN `train_nature.py` process on `ofi1`.
- No active SB3 process on `old1`.

The older DQN registry is mostly a historical ledger now, not a reliable source
of current activity.

The newest active intent appears to be the DeepMind reference control job in
`researchops/runs.d`, but it is blocked by `ofi2` being down and possibly by
Slurm account/QOS noise.

## Local Repo Map

Workspace:

- `/home/t/Downloads/dqn`

Important files:

- `PROJECT_STATUS.md`: CartPole history and first Slurm run.
- `cartpole/run.slurm`: early Slurm example.
- `atari/train_nature.py`: main scratch DQN trainer.
- `atari/eval.py`: scratch DQN evaluator.
- `atari/eval_checkpoint.py`: registry-aware evaluator and eval queue.
- `atari/auto_eval_milestones.py`: milestone/final auto-eval.
- `atari/research_status.py`: DQN research status renderer.
- `atari/research_telegram_command.sh`: command router for Telegram.
- `atari/telegram_notify_report.py`: Telegram report sender.
- `atari/run_registry.csv`: local registry copy.
- `atari/*.sbatch`: individual Slurm launch scripts.
- `atari/AUDIT_BREAKOUT.md`: detailed DQN/Breakout experiment audit.
- `atari/V3_PREFLIGHT.md`: preflight contract and interpretation thresholds.

The local worktree was already dirty before this report was added. I did not
revert or normalize unrelated changes.

## What Is Already Close To The Desired Future

Already present:

- Machine inventory.
- Slurm as a batch scheduler.
- Prometheus metrics.
- Telegram command interface.
- Telegram alerts.
- Per-run output directories.
- Metadata files.
- Rewards CSV.
- Diagnostics CSV.
- Checkpoint manifest.
- Eval ledger.
- Auto-eval queue.
- Final/best auto-eval summaries.
- Node adoption workflow.
- Project sync/run-check ideas in `poormansops`.

Missing or inconsistent:

- One blessed launcher.
- One run directory contract across all experiments.
- Fresh authoritative run registry.
- Timeout-safe status commands.
- Strong separation between machine health and experiment health.
- Exact resume support for scratch Atari DQN.
- A uniform metrics JSONL or tracker interface.
- Clean secrets handling.
- Slurm node-state repair/maintenance procedure.

## Recommended Next Moves

1. Freeze the current inventory and status commands in docs.
2. Repair node health separately from experiment standardization:
   - bring `ofi2` and `old2` back online or remove/drain them from active use;
   - inspect why `old1` is reachable but Slurm remains down;
   - resolve job 56's pending/account/node issue.
3. Mark stale DQN registry rows accurately:
   - completed runs should be `finished`;
   - dead/stale runs should not appear as active.
4. Add hard timeouts to all status paths that SSH to run owners.
5. Decide the run contract:

```text
runs/<project>/<run_id>/
  config.yaml
  metadata.json
  metrics.jsonl
  evals.jsonl
  checkpoints/
  artifacts/
  logs/
```

6. Build one launcher around existing pieces:

```text
submit_run --config configs/foo.yaml
```

7. Make Telegram read the run contract instead of per-experiment log formats.
8. Move remaining hard-coded secrets into private env files.
9. Decide whether Atari DQN must become resumable. If yes, checkpoint replay
   memory and handle Slurm stop signals. If no, label those runs non-resumable
   and rely on best/final artifacts.

## Bottom Line

You are not missing infrastructure. You have infrastructure, but it has grown in
layers:

- `poormansops` manages machines.
- Prometheus observes machines.
- Slurm owns job scheduling.
- DQN/researchops scripts interpret experiments.
- Telegram exposes status and alerts.

The cleanup is to standardize the boundary between experiment code and this
outer shell. Make every serious experiment become a run with a config, run id,
logs, metrics, artifacts, checkpoints, and status. Keep the internals loose, but
make the boundary strict.
