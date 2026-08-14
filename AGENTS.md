# Agent Operating Guide

## Purpose

This repository is part of a distributed research-compute system. The system is
not only the local training code; it also includes inventory-driven node
management, Slurm batch execution, Prometheus machine observability, run and
evaluation records, and Telegram status/alert surfaces.

Agents working in this repository must preserve those boundaries. A locally
reasonable edit can still be wrong if it creates a second scheduler, a second
run-state model, a second metrics format, or a Telegram command that bypasses
the normal sources of truth.

## Required Reading

Before modifying infrastructure, experiment execution, status reporting,
evaluation, launch scripts, registry behavior, or Telegram commands, read:

- [docs/SYSTEM.md](docs/SYSTEM.md)
- [docs/RUN_CONTRACT.md](docs/RUN_CONTRACT.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)

[CURRENT_SYSTEM_ARCHEOLOGY.md](CURRENT_SYSTEM_ARCHEOLOGY.md)
is a dated evidence report. Use it to understand what was observed during the
2026-08-06 archaeology pass, but treat the documents under `docs/` as the
canonical design and operating references.

## Architectural Boundaries

The system is divided into five domains:

1. Node inventory and provisioning.
2. Machine monitoring.
3. Job scheduling.
4. Experiment tracking and evaluation.
5. Remote reporting and control.

A change should normally belong to one domain. If a change crosses domains, the
interface between those domains must be made explicit in code and documentation.
For example, Slurm owns job scheduling, while Prometheus owns machine metrics;
experiment status code may read both, but it should not make either one pretend
to be the other.

The GitHub repositories also have different ownership boundaries:

| Repository | Owns |
|---|---|
| `thomaspradae/poormans` | Public Ubuntu node bootstrap only. |
| `thomaspradae/poormansops` | Private cluster control: inventory, adoption, admission, Slurm/Prometheus config, worker registry, attach/detach. |
| `thomaspradae/researchops` | Private experiment lifecycle: contracts, Slurm-backed submit/status/cancel/collect, run manifests, worker provenance. |
| `thomaspradae/dqn` | DQN/Atari experiment code and project-specific status/evaluation integration. |

Do not move private controller logic into the public bootstrap repository. Do
not move Slurm configuration or node admission into `researchops`. Do not add
generic cluster-control machinery to this DQN repository unless it is strictly a
project-specific adapter.

## Sources of Truth

Agents must identify the authoritative source before changing state. Derived
files and cached summaries must not be edited as if they were primary inputs.

| State | Authoritative source | Derived or consumer state |
|---|---|---|
| Node inventory | `/etc/poormans/nodes.env` on the controller | SSH blocks, Prometheus file SD, Slurm node fragments, dashboard aliases |
| Partition inventory | `/etc/poormans/partitions.env` on the controller | Slurm partition config |
| Machine metrics | Prometheus scraping `node_exporter` | `poormans` dashboard, status summaries, Telegram resource reports |
| Scheduler state | Slurm controller state | `squeue`, `sinfo`, Telegram job reports, status renderers |
| Worker identity/lifecycle | `/etc/poormans/workers.json` on the controller | Run worker snapshots, leased-worker history, status summaries |
| Experiment run state | Run directory, registry index, and evaluation ledger | Telegram summaries, status reports, plots |
| Evaluation state | Evaluation queue and evaluation result files | Auto-eval summaries, Telegram notifications |
| Alerts | Telegram configuration and watcher state files | Delivered chat messages |

When deployed state can differ from the local repository, inspect the deployed
state before changing behavior. Important deployed locations include
`/home/uace/poormansops`, `/home/uace/researchops`, `/etc/poormans`,
`/etc/prometheus`, `/etc/slurm`, and `/home/uace/dqn`.

## Change Procedure

Before changing code or operational behavior:

1. Inspect the current implementation and, when relevant, the deployed service
   or node state.
2. Identify which component owns the behavior.
3. Determine whether the change modifies an interface, file format, command
   contract, lifecycle state, or operational procedure.
4. Reuse the existing mechanism unless there is a concrete reason to replace
   it.
5. Validate through the supported operational path rather than only through a
   local unit of code.
6. Update the relevant documentation in the same change when behavior changes.

Do not store SSH passwords, sudo passwords, Telegram tokens, API keys, or other
secrets in repository files or documentation. If a secret is discovered in a
deployed script, document the cleanup need without copying the secret value.

## Experiment Integration

New experiment types must integrate through the run contract. Framework-specific
implementation details may remain internal to the experiment, but every serious
run must expose:

- run identity;
- immutable initial configuration;
- lifecycle state;
- logs;
- timestamped metrics;
- artifacts;
- evaluation records;
- checkpoint and resume semantics.

Do not create a separate tracking system for each framework. Existing DQN,
Stable Baselines3, DeepMind reference, scikit-learn, or generic PyTorch formats
may need adapters, but adapters should translate legacy output into the common
run model. They should not expand the common model around every legacy detail.

Real experiments should be submitted through the supported launcher and Slurm
path. Direct shell, `nohup`, or ad hoc `tmux` execution is acceptable for
debugging only when the run is clearly marked as non-canonical or is imported
through an adapter afterward.

SSH is not an experiment executor. SSH and Tailscale are transport, bootstrap,
inspection, and repair mechanisms used by `poormansops`. Permanent, borrowed,
and rented machines must become Slurm workers before normal runs use them. Do
not introduce `executor = ssh` as a peer of `executor = slurm` unless the
architecture is explicitly reopened because a real provider cannot join Slurm.

## Compatibility And Migration

The current system contains historical layers. Some scripts are active, some
are transitional, and some exist only for older experiments. Compatibility work
should make legacy records readable without making legacy behavior the default.

When introducing a new schema or command behavior, include a compatibility plan:
either preserve old readers, add an adapter, or write a migration. If a migration
is intentionally not provided, state which historical records are no longer
supported and why.

## Documentation Requirement

Changes to any of the following require documentation updates:

- node lifecycle or adoption behavior;
- worker registry, leased-worker attach/detach behavior, or provider metadata;
- Slurm configuration, partitions, submission behavior, or requeue semantics;
- Prometheus scrape topology or machine-health interpretation;
- run directory layout;
- run metadata, metrics, lifecycle, registry, or evaluation schemas;
- checkpoint naming, retention, or resume behavior;
- Telegram commands, routing, or alert behavior;
- status command behavior, especially SSH timeout behavior;
- operational procedures for repair, extension, deployment, or rollback.

If a code change has no documentation impact, say that explicitly in the final
engineering note or commit message. If it does have documentation impact, update
the relevant file in this repository as part of the same change.
