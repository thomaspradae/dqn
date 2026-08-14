# Experiment Run Contract

## Purpose

This document defines the framework-independent interface between experiment
code and the surrounding infrastructure. It is the contract that lets Slurm,
status commands, evaluation tools, Telegram commands, archives, and future
agents understand a run without knowing the internal details of every ML
framework.

Experiment code may be specialized. The run boundary must not be. A DQN trainer,
a Stable Baselines3 job, a DeepMind reference run, a scikit-learn baseline, and
a generic PyTorch experiment should all expose the same outer shape.

## Run Definition

A run is one bounded execution of an experiment. It has a unique identity,
immutable initial configuration, lifecycle state, logs, timestamped metrics,
artifacts, evaluation records, scheduler metadata when applicable, and declared
checkpoint behavior.

A run is not the same as a Slurm job. A run may have one scheduler job, no
scheduler job for imported historical data, or multiple jobs if a future
launcher supports requeue/restart. Slurm owns scheduling. The run contract owns
experiment evidence.

## Run Identity

Every run must have these identity fields:

| Field | Required | Meaning |
|---|---:|---|
| `contract_version` | yes | Version of this run contract used by the run. |
| `run_id` | yes | Stable unique id within a project. |
| `project` | yes | Project namespace, for example `atari`. |
| `experiment_type` | yes | Adapter or framework type, for example `scratch_dqn`, `sb3`, `deepmind_torch7`, `pytorch`, or `sklearn`. |
| `created_at` | yes | UTC timestamp when run identity was created. |
| `submitted_by` | yes | User or system that submitted/imported the run. |
| `code_version` | yes | Git commit, code hash set, external repo revision, or explicit `unknown`. |
| `scheduler_job_id` | conditional | Slurm job id when scheduled through Slurm. |
| `owner_node` | conditional | Node expected to own active process or primary artifacts. |

`run_id` should be lowercase ASCII and should avoid spaces. Use stable names
that encode the experimental intent only when useful, for example
`breakout_v4_seed20260621_dmrmsprop_bilinear`. Do not encode transient state
such as current status into `run_id`.

## Directory Layout

The target directory layout is:

```text
runs/<project>/<run_id>/
  config.yaml
  metadata.json
  state.json
  metrics.jsonl
  evals.jsonl
  checkpoints/
  artifacts/
  logs/
```

Path semantics:

| Path | Requirement | Mutability | Purpose |
|---|---|---|---|
| `config.yaml` | required | immutable after submission | Original or resolved run configuration. |
| `metadata.json` | required | append-safe updates only for discovered metadata | Identity, environment, scheduler, code, and resume metadata. |
| `state.json` | required | mutable by lifecycle owner | Current lifecycle state and last update. |
| `metrics.jsonl` | required for active runs | append-only | Training, system, throughput, optimizer, and custom metrics. |
| `evals.jsonl` | required when evals exist | append-only | Evaluation requests and results. |
| `checkpoints/` | required if checkpointing exists | append-only except retention policy | Model and state checkpoints plus manifest. |
| `artifacts/` | optional | append-only | Plots, reports, exports, videos, summaries, or derived outputs. |
| `logs/` | required | append-only or rotated | Scheduler, stdout, stderr, and experiment logs. |

Legacy DQN directories currently use files such as `RUN_METADATA.json`,
`run_metadata.json`, `rewards.csv`, `training_diagnostics.csv`,
`checkpoints.csv`, `eval_results.csv`, `q_net_ep*.pt`,
`checkpoint_ep*.pt`, `q_net_final.pt`, and `checkpoint_final.pt`. Those files
are valid legacy evidence. Adapters should map them into this contract rather
than requiring old runs to be rewritten in place.

## Configuration

Each run must preserve the configuration used to start it. Configuration should
include user-specified values, resolved defaults, command-line overrides,
environment-dependent values that affect behavior, and resource requests.

The configuration should be hashable. A launcher should be able to compute a
configuration hash after defaults and overrides are resolved. The hash is used
for auditability, not as the run id.

Configuration is immutable after submission. If a run is restarted with a
different configuration, it is a different run unless the launcher explicitly
models restart attempts under one run identity.

## Metadata Schema

`metadata.json` must include enough information for a future operator to
understand what code ran, where it ran, and what state was captured.

Required top-level fields:

```json
{
  "contract_version": "1.0",
  "run_id": "example_run",
  "project": "atari",
  "experiment_type": "scratch_dqn",
  "created_at": "2026-08-06T00:00:00Z",
  "submitted_by": "uace",
  "command": "python train_nature.py ...",
  "argv": ["python", "train_nature.py"],
  "code_version": {
    "git_commit": "unknown",
    "dirty": null,
    "code_hashes": {}
  },
  "environment": {
    "python": "unknown",
    "packages": {},
    "cuda": null,
    "device": null
  },
  "scheduler": {
    "system": "slurm",
    "job_id": null,
    "partition": null,
    "requested_node": null
  },
  "host": {
    "owner_node": null,
    "hostname": null
  },
  "randomness": {
    "seed": null
  },
  "resume": {
    "category": "non_resumable",
    "supported": false,
    "reason": "not declared"
  }
}
```

Additional framework-specific metadata is allowed, but generic tools must be
able to ignore it safely.

Recommended metadata fields include optimizer settings, neural network
architecture summary, dataset or environment id, package versions, CUDA driver
and runtime information, CPU/GPU device information, Slurm resource request,
signal-handling policy, checkpoint policy, and retention policy.

## Lifecycle State

Allowed lifecycle states are:

```text
created
submitted
pending
running
stopping
completed
failed
cancelled
interrupted
archived
```

State meanings:

| State | Meaning |
|---|---|
| `created` | Run identity and directory exist, but submission has not occurred. |
| `submitted` | Launcher submitted the run to a scheduler or remote executor. |
| `pending` | Scheduler has accepted the job but has not started it. |
| `running` | The experiment process is believed to be active. |
| `stopping` | A controlled stop has been requested. |
| `completed` | The run finished successfully according to experiment code. |
| `failed` | The run ended with an error or failed validation. |
| `cancelled` | The operator or scheduler intentionally cancelled the run. |
| `interrupted` | The run stopped before completion and may or may not be resumable. |
| `archived` | The run is no longer active but preserved for historical lookup. |

The launcher may write `created`, `submitted`, and initial `pending`. Slurm
observation may update `pending`, `running`, `cancelled`, or `failed` when the
scheduler has clear evidence. Experiment code may write `running`, `completed`,
`failed`, and heartbeat fields. Evaluation code must not mark a training run
`completed`; it only writes evaluation state.

`state.json` should include:

```json
{
  "state": "running",
  "updated_at": "2026-08-06T00:00:00Z",
  "source": "trainer",
  "message": "episode 1200",
  "attempt": 1
}
```

## Metrics

`metrics.jsonl` is append-only. Each line is one JSON event.

Common event structure:

```json
{
  "timestamp": "2026-08-06T00:00:00Z",
  "step": 100000,
  "step_unit": "agent_step",
  "namespace": "train",
  "name": "loss",
  "value": 0.031
}
```

This format intentionally supports arbitrary metrics. Infrastructure does not
need to understand the meaning of every metric; it only needs a stable event
shape.

Metric namespace conventions:

| Namespace | Use |
|---|---|
| `train` | Training returns, losses, exploration, episode counts, environment progress. |
| `eval` | Evaluation returns, episode lengths, success rates, aggregate scores. |
| `system` | Process-level runtime signals captured by experiment code. |
| `throughput` | Steps per second, frames per second, samples per second. |
| `optimizer` | Learning rate, gradient norm, optimizer-specific values. |
| `data` | Dataset, replay, loader, or sampling metrics. |

Legacy CSV metrics are acceptable through adapters. For current scratch DQN,
`rewards.csv` and `training_diagnostics.csv` should be treated as legacy metric
sources until the trainer writes `metrics.jsonl` directly.

## Logging

Each run must preserve logs under `logs/` or map legacy log paths into the run
record.

Required log classes:

- scheduler stdout and stderr;
- experiment stdout and stderr when separate;
- structured events if the framework emits them;
- evaluation logs for evaluations associated with the run.

Logs should use UTC timestamps when generated by system code. Slurm output paths
must be recorded in metadata so a status tool can find logs even when they are
outside the run directory.

## Checkpoints

Checkpoint behavior must be explicit. A run that writes model weights but cannot
resume exact training must say so.

Recommended checkpoint layout:

```text
checkpoints/
  manifest.jsonl
  latest
  best
  final
  checkpoint_step000100000.pt
```

`manifest.jsonl` should be append-only and include checkpoint path, timestamp,
step, step unit, episode when applicable, type, metric summary, and whether the
checkpoint is sufficient for resume.

Checkpoint writes should be atomic: write to a temporary path, flush when
possible, then rename into place. Symlinks or pointer files for `latest`, `best`,
and `final` are acceptable if readers handle missing targets cleanly.

Retention policy must be declared. If old checkpoints are deleted, the manifest
must either preserve tombstones or clearly show that the artifact was removed by
policy.

## Resume Semantics

Every run must declare one of:

| Category | Meaning |
|---|---|
| `fully_resumable` | Exact training can continue with model, optimizer, counters, random state, data/replay state, and environment-critical state restored. |
| `partially_resumable` | Some training state can continue, but learning is not bitwise or algorithmically exact. Missing state must be documented. |
| `non_resumable` | Checkpoints are useful for evaluation or export, but training continuation is not supported. |

For scratch Atari DQN, exact resume requires at least model weights, target
network, optimizer state, training counters, environment counters, random states,
and replay memory. The current `train_nature.py` explicitly reports that replay
buffer checkpointing is false and resume support is false. Therefore current
scratch Atari DQN runs are non-resumable even if Slurm can requeue the process.

Future resumable trainers should handle scheduler stop signals, flush state
before termination, write a checkpoint manifest entry, and update `state.json`
before exiting.

## Evaluation

Evaluation is a first-class record attached to a run. It is not only a log
message.

Evaluation request fields:

| Field | Required | Meaning |
|---|---:|---|
| `eval_id` | yes | Unique id for this evaluation request. |
| `run_id` | yes | Training run being evaluated. |
| `checkpoint_selector` | yes | `latest`, `final`, `best`, explicit step, episode, or path. |
| `checkpoint_path` | conditional | Resolved path when known. |
| `episodes` or `samples` | yes | Evaluation budget. |
| `seed_policy` | yes | Fixed, random, inherited, or deterministic policy. |
| `status` | yes | `queued`, `running`, `completed`, `failed`, or `superseded`. |
| `metrics` | conditional | Result metrics when completed. |

Recommended result event:

```json
{
  "timestamp": "2026-08-06T00:00:00Z",
  "eval_id": "example_run_final_30ep",
  "run_id": "example_run",
  "checkpoint_selector": "final",
  "checkpoint_path": "checkpoints/final",
  "episodes": 30,
  "seed_policy": "fixed",
  "status": "completed",
  "metrics": {
    "eval/mean_return": 49.0,
    "eval/min_return": 12.0,
    "eval/max_return": 87.0
  }
}
```

Current DQN evaluation uses `eval_checkpoint.py`, `eval_queue.csv`,
`eval_results.csv`, `auto_eval_state.csv`, and auto-eval summaries. Those are
valid current state files. The target contract makes their content portable
across experiment types.

## Scheduler Integration

The scheduler contract is between the launcher and Slurm. The launcher provides
resource requests, partition or node constraints, time limit, working directory,
environment, output paths, and command. Slurm returns or exposes job id, pending
state, start time, node allocation, exit state, and logs.

Run metadata should record:

- Slurm job id;
- partition;
- requested nodes or constraints;
- allocated nodes when known;
- CPU, memory, GPU, and time requests;
- stdout and stderr paths;
- requeue policy;
- signal policy;
- submit time and start time when known.

A real run should not rely only on a process id. Process ids are node-local and
ephemeral. Slurm job id plus run id plus run directory is the durable identity
set.

## Registry

The registry is an index, not the sole source of truth. It exists for discovery,
querying, filtering, and human overview. Detailed evidence belongs in the run
directory and scheduler state belongs in Slurm.

Registry rows should include:

- `run_id`;
- `project`;
- `experiment_type`;
- `owner_node`;
- `scheduler_job_id`;
- `run_dir`;
- `status`;
- `created_at`;
- `updated_at`;
- `target_steps` or equivalent budget;
- adapter type;
- notes.

If registry status conflicts with run-directory evidence or Slurm, status tools
should surface the conflict and avoid silently presenting stale data as live
truth.

## Framework Adapters

Adapters translate framework-specific evidence into the common run model.

Initial adapter targets:

| Adapter | Current evidence |
|---|---|
| `scratch_dqn` | DQN metadata JSON, reward CSV, diagnostics CSV, checkpoint CSV, eval CSV, PyTorch checkpoints. |
| `sb3` | Monitor CSV, SB3 zip files, SB3 logs, final model artifacts. |
| `deepmind_torch7` | `researchops/runs.d` entries, DeepMind reference logs, checkpoint files. |
| `pytorch` | Generic metadata, metrics JSONL or TensorBoard adapter, checkpoints. |
| `sklearn` | Config, metrics, serialized estimator artifacts, reports. |

An adapter should be read-oriented unless a migration is explicitly planned.
Adapters may emit normalized status or metrics, but they should not mutate
historical run directories by default.

## Contract Versioning

Every run must record `contract_version`. This document defines version `1.0`.

Backward-compatible changes may add optional fields. Breaking changes require a
new contract version and either a migration or an adapter. Status tools should
refuse to guess when a run has an unknown contract version; they should report
the unsupported version and continue with other runs.

Legacy runs without `contract_version` should be treated as imported legacy
records and mapped through an adapter.

## Minimum Compliance Checklist

A new serious experiment type is contract-compliant when it can answer these
questions without reading framework internals:

- What run is this?
- What configuration started it?
- Which code version produced it?
- Was it scheduled, where, and under which job id?
- Is it pending, running, completed, failed, cancelled, interrupted, or archived?
- Where are its logs?
- Where are its metrics?
- Where are its artifacts?
- Which checkpoints exist?
- Can training resume exactly, partially, or not at all?
- Which evaluations were requested and what were their results?

If any answer is unavailable, the run must either add the missing record or mark
the field explicitly unknown.
