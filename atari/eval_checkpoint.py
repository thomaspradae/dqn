#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_status import (
    ShellRunner,
    actual_eval_process_counts,
    canonical_registry_label,
    checkpoint_glob,
    checkpoint_path_for_episode,
    choose_prom,
    fmt_percent,
    load_cluster_resources,
    load_registry,
    parse_checkpoint_manifest,
    parse_eval_results,
    parse_log_progress,
    parse_rewards,
    resolve_run,
    safe_float,
    safe_int,
    select_active_checkpoint_episode,
)


EVAL_FIELDS = [
    "timestamp",
    "run_id",
    "game",
    "env_id",
    "node",
    "owner_node",
    "checkpoint_id",
    "checkpoint_path",
    "checkpoint",
    "checkpoint_episode",
    "model",
    "episodes",
    "epsilon",
    "frame_skip",
    "noop_max",
    "max_steps_per_episode",
    "threads",
    "mean",
    "std",
    "min",
    "max",
    "host",
    "status",
    "log",
    "command",
    "requested_by",
    "trigger",
    "milestone_step",
]

QUEUE_FIELDS = [
    "created_at",
    "updated_at",
    "request_id",
    "run_id",
    "checkpoint_selector",
    "episodes",
    "epsilon",
    "requested_by",
    "trigger",
    "milestone_step",
    "status",
    "assigned_node",
    "result_path",
    "attempts",
    "last_error",
]

CANONICAL_NODE = {
    "run_id": "__canonical__",
    "node": "ofi1",
    "owner_node": "ofi1",
    "ssh_target": "uace@100.107.98.78",
}
QUEUE_PATH = os.environ.get("DQN_RESEARCH_QUEUE_PATH", "/home/uace/dqn/atari/eval_queue.csv")
STALE_RUNNING_SECONDS = 30 * 60
TERMINAL_EVAL_FAILURE_PATTERNS = [
    "unrecognized arguments:",
    "no such file or directory",
    "does not exist",
    "filenotfounderror",
    "modulenotfounderror",
    "importerror",
    "invalid choice:",
    "missing checkpoint:",
]


def ssh_base() -> list[str]:
    return shlex.split(
        os.environ.get(
            "DQN_RESEARCH_SSH",
            "ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new",
        )
    )


def run_remote_python(runner: ShellRunner, run: dict[str, str], script: str, args: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    if runner.is_local_node(run.get("owner_node") or run["node"]):
        cmd = ["python3", "-c", script] + args
    else:
        target = runner.target_for(run)
        remote_cmd = " ".join(shlex.quote(part) for part in ["python3", "-c", script] + args)
        cmd = ssh_base() + [target, remote_cmd]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)


def remote_file_exists(runner: ShellRunner, run: dict[str, str], path: str) -> bool:
    result = runner.run_shell(run, f"test -f {shlex.quote(path)} && printf yes || true", 10)
    return result.stdout.strip() == "yes"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_remote_file(runner: ShellRunner, run: dict[str, str], path: str, timeout: int = 20) -> str:
    result = runner.run_shell(run, f"test -r {shlex.quote(path)} && cat {shlex.quote(path)} || true", timeout)
    return result.stdout


def read_queue(runner: ShellRunner) -> list[dict[str, str]]:
    text = read_remote_file(runner, CANONICAL_NODE, QUEUE_PATH, 15)
    if not text.strip():
        return []
    rows = []
    for row in csv.DictReader(text.splitlines()):
        cleaned = {field: (row.get(field) or "") for field in QUEUE_FIELDS}
        if cleaned.get("request_id"):
            rows.append(cleaned)
    return rows


def write_queue(runner: ShellRunner, rows: list[dict[str, str]]) -> None:
    payload = json.dumps([{field: row.get(field, "") for field in QUEUE_FIELDS} for row in rows])
    script = (
        "import csv,json,os,sys;"
        "path=sys.argv[1]; fields=json.loads(sys.argv[2]); rows=json.loads(sys.argv[3]);"
        "os.makedirs(os.path.dirname(path), exist_ok=True);"
        "tmp=path+'.tmp';"
        "f=open(tmp,'w',newline='');"
        "w=csv.DictWriter(f,fieldnames=fields);"
        "w.writeheader();"
        "w.writerows(rows);"
        "f.close();"
        "os.replace(tmp,path)"
    )
    result = run_remote_python(runner, CANONICAL_NODE, script, [QUEUE_PATH, json.dumps(QUEUE_FIELDS), payload], 20)
    if result.returncode != 0:
        raise SystemExit(f"failed to write eval queue: {result.stderr.strip()}")


def enqueue_eval(
    runner: ShellRunner,
    run_id: str,
    checkpoint_selector: str,
    episodes: int,
    epsilon: float,
    requested_by: str,
    trigger: str,
    milestone_step: str = "",
) -> dict[str, str]:
    now = utc_stamp()
    row = {
        "created_at": now,
        "updated_at": now,
        "request_id": uuid.uuid4().hex[:12],
        "run_id": run_id,
        "checkpoint_selector": checkpoint_selector,
        "episodes": str(episodes),
        "epsilon": str(epsilon),
        "requested_by": requested_by,
        "trigger": trigger,
        "milestone_step": milestone_step,
        "status": "pending",
        "assigned_node": "",
        "result_path": "",
        "attempts": "0",
        "last_error": "",
    }
    rows = read_queue(runner)
    duplicate = [
        existing
        for existing in rows
        if existing.get("run_id") == row["run_id"]
        and existing.get("checkpoint_selector") == row["checkpoint_selector"]
        and existing.get("episodes") == row["episodes"]
        and existing.get("status") in {"pending", "running"}
        and existing.get("trigger") == row["trigger"]
        and existing.get("milestone_step") == row["milestone_step"]
    ]
    if duplicate:
        return duplicate[0]
    rows.append(row)
    write_queue(runner, rows)
    return row


def render_queue(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "Eval queue: empty"
    lines = ["Eval queue:"]
    for row in rows:
        lines.append(
            f"{row['request_id']} {row['status']} {row['run_id']} "
            f"{row['checkpoint_selector']} {row['episodes']}ep "
            f"trigger={row.get('trigger') or 'manual'} attempts={row.get('attempts') or '0'}"
        )
        if row.get("last_error"):
            lines.append(f"  last_error: {row['last_error'][:180]}")
    return "\n".join(lines)


def is_terminal_eval_failure(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in TERMINAL_EVAL_FAILURE_PATTERNS)


def queue_stamp_age_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        stamp = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - stamp).total_seconds()


def active_eval_node_labels(runs: list[dict[str, str]]) -> set[str]:
    labels = {
        (run.get("owner_node") or run.get("node") or "").strip()
        for run in runs
        if run.get("type") == "scratch_dqn"
    }
    labels.add("ofi1")
    labels.discard("")
    return labels


def list_dqn_checkpoints(runner: ShellRunner, run: dict[str, str]) -> list[int]:
    run_dir = shlex.quote(run["run_dir"])
    pattern = checkpoint_glob(run)
    result = runner.run_shell(
        run,
        f"find {run_dir} -maxdepth 1 -type f -name {shlex.quote(pattern)} -print 2>/dev/null || true",
        20,
    )
    episodes = []
    for line in result.stdout.splitlines():
        match = re.search(r"q_net_ep(\d+)\.pt$", line.strip())
        if match:
            episodes.append(int(match.group(1)))
    return sorted(episodes)


def current_training_episode(runner: ShellRunner, run: dict[str, str]) -> int | None:
    log_tail = runner.tail_file(run, run["log"], 1000, 20)
    progress = parse_log_progress(log_tail)
    candidates = []
    if progress.get("episode") is not None:
        candidates.append(int(progress["episode"]))
    rewards = parse_rewards(read_remote_file(runner, run, f"{run['run_dir']}/rewards.csv", 20))
    if rewards.get("latest_episode") is not None:
        candidates.append(int(rewards["latest_episode"]))
    return max(candidates) if candidates else None


def latest_active_checkpoint_episode(runner: ShellRunner, run: dict[str, str]) -> int | None:
    episodes = list_dqn_checkpoints(runner, run)
    checkpoints = {"episodes": episodes, "latest_episode": max(episodes) if episodes else None}
    manifest = parse_checkpoint_manifest(read_remote_file(runner, run, f"{run['run_dir']}/checkpoints.csv", 20))
    current_episode = current_training_episode(runner, run)
    latest, _, _ = select_active_checkpoint_episode(checkpoints, manifest, current_episode)
    return latest


def nearest_checkpoint_episode(episodes: list[int], target_episode: int) -> int | None:
    if not episodes:
        return None
    return min(episodes, key=lambda ep: (abs(ep - target_episode), ep > target_episode, ep))


def resolve_best_train_checkpoint(runner: ShellRunner, run: dict[str, str]) -> tuple[str, str, str]:
    rewards_text = read_remote_file(runner, run, f"{run['run_dir']}/rewards.csv", 30)
    rewards = parse_rewards(rewards_text)
    best_episode = rewards.get("best_last100_episode")
    if best_episode is None:
        raise SystemExit(f"no best_train available for {run['run_id']}: not enough rewards")
    checkpoints = list_dqn_checkpoints(runner, run)
    nearest = nearest_checkpoint_episode(checkpoints, int(best_episode))
    if nearest is None:
        raise SystemExit(f"no checkpoints available for {run['run_id']}")
    path = checkpoint_path_for_episode(run, nearest)
    if path is None:
        raise SystemExit(f"cannot resolve checkpoint pattern for {run['run_id']}")
    return f"best_train_ep{nearest}", str(nearest), path


def resolve_best_eval_checkpoint(runner: ShellRunner, run: dict[str, str]) -> tuple[str, str, str]:
    eval_text = read_remote_file(runner, run, f"{run['run_dir']}/eval_results.csv", 20)
    evals = parse_eval_results(eval_text)
    if evals.get("count", 0) == 0:
        raise SystemExit(
            f"No eval results yet for {run['run_id']}. "
            f"Try /eval {run['run_id']} best_train 10 or /eval {run['run_id']} latest 10."
        )
    best = evals["best"]
    checkpoint_path = best.get("checkpoint_path") or best.get("model")
    checkpoint_id = best.get("checkpoint_id") or best.get("checkpoint") or Path(checkpoint_path or "").stem
    checkpoint_episode = best.get("checkpoint_episode", "")
    if not checkpoint_path:
        raise SystemExit(f"best_eval row for {run['run_id']} has no checkpoint_path")
    if not remote_file_exists(runner, run, checkpoint_path):
        raise SystemExit(f"best_eval checkpoint is missing: {checkpoint_path}")
    return checkpoint_id or "best_eval", checkpoint_episode, checkpoint_path


def resolve_checkpoint(runner: ShellRunner, run: dict[str, str], checkpoint: str) -> tuple[str, str, str]:
    token = checkpoint.strip()
    if token == "latest":
        # A completed scratch-DQN run writes q_net_final.pt at an episode that
        # is usually not a checkpoint cadence boundary.  Prefer that artifact
        # over reconstructing q_net_ep<last_episode>.pt, which does not exist
        # for a normal final save.
        final_path = f"{run['run_dir']}/q_net_final.pt"
        if remote_file_exists(runner, run, final_path):
            return "final", "final", final_path
        ep = latest_active_checkpoint_episode(runner, run)
        if ep is None:
            raise SystemExit(f"no checkpoint matching {checkpoint_glob(run)} found for {run['run_id']}")
        path = checkpoint_path_for_episode(run, ep)
        if path is None:
            raise SystemExit(f"cannot resolve checkpoint pattern for {run['run_id']}")
        return f"ep{ep}", str(ep), path
    if token == "final":
        path = f"{run['run_dir']}/q_net_final.pt"
        if not remote_file_exists(runner, run, path):
            raise SystemExit(f"missing checkpoint: {path}")
        return "final", "final", path
    if token in {"best", "best_train"}:
        return resolve_best_train_checkpoint(runner, run)
    if token == "best_eval":
        return resolve_best_eval_checkpoint(runner, run)
    match = re.fullmatch(r"ep?(\d+)", token)
    if match:
        ep = int(match.group(1))
        path = checkpoint_path_for_episode(run, ep)
        if path is None:
            raise SystemExit(f"cannot resolve checkpoint pattern for {run['run_id']}")
        if not remote_file_exists(runner, run, path):
            raise SystemExit(f"missing checkpoint: {path}")
        return f"ep{ep}", str(ep), path
    if token.startswith("/"):
        if not remote_file_exists(runner, run, token):
            raise SystemExit(f"missing checkpoint: {token}")
        return Path(token).stem, "", token
    raise SystemExit(f"unsupported checkpoint selector: {checkpoint}")


def resource_ok(run: dict[str, str], max_cpu: float, max_ram: float) -> tuple[bool, str]:
    prom = choose_prom()
    resources = load_cluster_resources(prom)
    owner_node = run.get("owner_node") or run["node"]
    node_row = resources.get(owner_node, {})
    cpu = node_row.get("cpu")
    ram = node_row.get("ram")
    if cpu is None or ram is None:
        return True, "resources unknown"
    ok = cpu <= max_cpu and ram <= max_ram
    msg = f"{owner_node} cpu={fmt_percent(cpu)} ram={fmt_percent(ram)} thresholds={max_cpu:.1f}/{max_ram:.1f}"
    return ok, msg


def parse_eval_output(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for key in ("episodes", "mean", "std", "min", "max"):
        match = re.search(rf"^{key}=([-0-9.]+)", text, re.MULTILINE)
        if match:
            parsed[key] = match.group(1)
    if "mean" not in parsed:
        rewards = [safe_float(x) for x in re.findall(r"reward=([-0-9.]+)", text)]
        rewards = [r for r in rewards if r is not None]
        if rewards:
            mean = sum(rewards) / len(rewards)
            var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            parsed.update(
                {
                    "episodes": str(len(rewards)),
                    "mean": f"{mean:.2f}",
                    "std": f"{var ** 0.5:.2f}",
                    "min": f"{min(rewards):.2f}",
                    "max": f"{max(rewards):.2f}",
                }
            )
    return parsed


def append_eval_result(runner: ShellRunner, run: dict[str, str], row: dict[str, Any]) -> None:
    script = (
        "import csv,json,os,sys;"
        "path=sys.argv[1]; fields=json.loads(sys.argv[2]); row=json.loads(sys.argv[3]);"
        "os.makedirs(os.path.dirname(path), exist_ok=True);"
        "write_header=(not os.path.exists(path)) or os.path.getsize(path)==0;"
        "f=open(path,'a',newline='');"
        "w=csv.DictWriter(f,fieldnames=fields);"
        "w.writeheader() if write_header else None;"
        "w.writerow(row);"
        "f.close()"
    )
    result = run_remote_python(
        runner,
        run,
        script,
        [f"{run['run_dir']}/eval_results.csv", json.dumps(EVAL_FIELDS), json.dumps(row)],
        20,
    )
    if result.returncode != 0:
        raise SystemExit(f"failed to append eval_results.csv: {result.stderr.strip()}")


def run_eval(runner: ShellRunner, run: dict[str, str], model_path: str, log_path: str, args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    venv_path = run.get("venv_path") or run.get("venv")
    eval_script = run.get("eval_script") or "eval.py"
    if not venv_path:
        raise SystemExit(f"run {run['run_id']} has no venv_path configured")
    if not eval_script:
        raise SystemExit(f"run {run['run_id']} has no eval_script configured")
    resize_interpolation = resize_interpolation_for_run(runner, run)
    activate = f"{venv_path}/bin/activate"
    command_parts = [
        "python",
        eval_script,
        "--env-id",
        run["env_id"],
        "--model",
        model_path,
        "--episodes",
        str(args.episodes),
        "--epsilon",
        str(args.epsilon),
        "--resize-interpolation",
        resize_interpolation,
        "--frame-skip",
        str(args.frame_skip),
        "--noop-max",
        str(args.noop_max),
        "--max-steps-per-episode",
        str(args.max_steps_per_episode),
        "--threads",
        str(args.threads),
    ]
    eval_cmd = " ".join(shlex.quote(part) for part in command_parts)
    remote = (
        "set -euo pipefail; "
        f"cd {shlex.quote(run['project_dir'])}; "
        f"source {shlex.quote(activate)}; "
        f"mkdir -p {shlex.quote(str(Path(log_path).parent))}; "
        "LOWPRI='nice -n 10'; "
        "if command -v ionice >/dev/null 2>&1; then LOWPRI='ionice -c2 -n7 nice -n 10'; fi; "
        f"$LOWPRI {eval_cmd} 2>&1 | tee {shlex.quote(log_path)}"
    )
    if args.dry_run:
        print(remote)
        return subprocess.CompletedProcess([], 0, "", "")
    if runner.is_local_node(run.get("owner_node") or run["node"]):
        cmd = ["bash", "-lc", remote]
    else:
        cmd = ssh_base() + [runner.target_for(run), "bash -lc " + shlex.quote(remote)]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def resize_interpolation_for_run(runner: ShellRunner, run: dict[str, str]) -> str:
    metadata = read_remote_file(runner, run, f"{run['run_dir']}/RUN_METADATA.json", 20)
    try:
        value = json.loads(metadata).get("resize_interpolation", "area")
    except (TypeError, ValueError):
        return "area"
    return value if value in {"area", "bilinear"} else "area"


def run_eval_request(
    runs: list[dict[str, str]],
    row: dict[str, str],
    force: bool = False,
) -> tuple[bool, str, str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        row["run_id"],
        row["checkpoint_selector"],
        row["episodes"],
        "--epsilon",
        row.get("epsilon") or "0.05",
        "--requested-by",
        row.get("requested_by") or "queue",
        "--trigger",
        row.get("trigger") or "queued",
        "--milestone-step",
        row.get("milestone_step") or "",
    ]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    text = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
    match = re.search(r"eval log:\s*(\S+)", text)
    result_path = match.group(1) if match else ""
    return proc.returncode == 0, text, result_path


def drain_queue(limit: int = 1, force: bool = False) -> None:
    runner = ShellRunner()
    rows = read_queue(runner)
    if not rows:
        print("Eval queue: empty")
        return

    runs = load_registry(None)
    eval_processes = actual_eval_process_counts(active_eval_node_labels(runs))
    live_eval_processes = sum(eval_processes.values())
    output_lines = []
    changed = False
    for row in rows:
        if row.get("status") == "running" and is_terminal_eval_failure(row.get("last_error", "")):
            row["status"] = "failed"
            row["updated_at"] = utc_stamp()
            output_lines.append(f"failed_terminal {row['request_id']}: {row.get('last_error', '')[:180]}")
            changed = True
            continue
        if row.get("status") == "running" and live_eval_processes == 0:
            age = queue_stamp_age_seconds(row.get("updated_at", ""))
            if age is not None and age > STALE_RUNNING_SECONDS:
                row["status"] = "failed"
                row["updated_at"] = utc_stamp()
                row["last_error"] = (
                    "stale_running: no eval process exists after >30 minutes; "
                    "marked failed so a fresh request can be queued"
                )
                output_lines.append(f"failed_stale {row['request_id']}: {row['last_error']}")
                changed = True
    if changed:
        write_queue(runner, rows)

    pending = [row for row in rows if row.get("status") == "pending"]
    if not pending:
        print("\n".join(output_lines) if output_lines else render_queue(rows))
        return

    processed = 0
    for row in pending:
        if processed >= limit:
            break
        row["status"] = "running"
        row["updated_at"] = utc_stamp()
        row["attempts"] = str(safe_int(row.get("attempts"), 0) + 1)
        write_queue(runner, rows)

        ok, text, result_path = run_eval_request(runs, row, force=force)
        row["updated_at"] = utc_stamp()
        if ok:
            row["status"] = "done"
            row["result_path"] = result_path
            row["last_error"] = ""
            processed += 1
            output_lines.append(f"done {row['request_id']} {row['run_id']} {row['checkpoint_selector']}")
        else:
            row["last_error"] = text[-500:]
            terminal_failure = is_terminal_eval_failure(text)
            row["status"] = "failed" if terminal_failure else "pending"
            state = "failed_terminal" if terminal_failure else "kept pending"
            output_lines.append(f"{state} {row['request_id']}: {row['last_error'][:180]}")
            if "refusing eval while node is busy" in text:
                break
        write_queue(runner, rows)

    print("\n".join(output_lines) if output_lines else "No eval requests drained")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"queue", "list-queue"}:
        runner = ShellRunner()
        print(render_queue(read_queue(runner)))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "drain":
        drain_parser = argparse.ArgumentParser()
        drain_parser.add_argument("drain")
        drain_parser.add_argument("--limit", type=int, default=1)
        drain_parser.add_argument("--force", action="store_true")
        drain_args = drain_parser.parse_args()
        drain_queue(drain_args.limit, drain_args.force)
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("checkpoint", nargs="?", default="latest")
    parser.add_argument("episodes_pos", nargs="?", type=int)
    parser.add_argument("mode_flags", nargs="*")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--max-steps-per-episode", type=int, default=4500)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-cpu", type=float, default=98.0)
    parser.add_argument("--max-ram", type=float, default=98.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--requested-by", default="telegram")
    parser.add_argument("--trigger", default="manual")
    parser.add_argument("--milestone-step", default="")
    args = parser.parse_args()

    flags = {flag.lower() for flag in args.mode_flags}
    if "force" in flags:
        args.force = True
    if "queue" in flags or "queued" in flags:
        args.queue = True
    if "strict" in flags:
        args.strict = True
    if args.strict:
        args.max_cpu = min(args.max_cpu, 85.0)
        args.max_ram = min(args.max_ram, 90.0)

    if args.episodes_pos is not None:
        args.episodes = args.episodes_pos
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")

    runs = load_registry(args.registry)
    run = resolve_run(runs, args.run)
    if run.get("type") != "scratch_dqn":
        if not run.get("eval_script"):
            raise SystemExit(
                f"run {run['run_id']} has no eval_script configured; "
                f"{run.get('algo', run.get('type'))} eval needs a dedicated evaluator"
            )
        raise SystemExit(f"eval_checkpoint currently supports scratch_dqn runs, got {run.get('type')}")

    runner = ShellRunner()
    checkpoint_label, checkpoint_episode, model_path = resolve_checkpoint(runner, run, args.checkpoint)

    ok, resource_msg = resource_ok(run, args.max_cpu, args.max_ram)
    if not ok and not args.force:
        if args.queue:
            row = enqueue_eval(
                runner,
                run["run_id"],
                args.checkpoint,
                args.episodes,
                args.epsilon,
                args.requested_by,
                args.trigger,
                str(args.milestone_step or ""),
            )
            print(f"queued eval: {row['request_id']}")
            print(f"run: {row['run_id']} checkpoint={row['checkpoint_selector']} episodes={row['episodes']}")
            print(f"resource check: {resource_msg}")
            print(f"queue: {QUEUE_PATH}")
            return
        raise SystemExit(f"refusing eval while node is busy: {resource_msg}. Use --force to override.")

    timestamp = utc_stamp()
    log_path = f"{run['run_dir']}/evals/{run['run_id']}_{checkpoint_label}_{args.episodes}ep_{timestamp}.log"
    host_result = runner.run_shell(run, "hostname", 10)
    host = host_result.stdout.strip().splitlines()[0] if host_result.stdout.strip() else run.get("owner_node", run["node"])
    proc = run_eval(runner, run, model_path, log_path, args)
    output = (proc.stdout or "") + (proc.stderr or "")
    parsed = parse_eval_output(output)
    if args.dry_run:
        status = "dry-run"
    else:
        status = "ok" if proc.returncode == 0 and "mean" in parsed else "failed"

    resize_interpolation = resize_interpolation_for_run(runner, run)
    command = (
        f"python {run.get('eval_script') or 'eval.py'} --env-id {run['env_id']} --model {model_path} --episodes {args.episodes} "
        f"--epsilon {args.epsilon} --resize-interpolation {resize_interpolation} --frame-skip {args.frame_skip} --noop-max {args.noop_max} "
        f"--max-steps-per-episode {args.max_steps_per_episode} --threads {args.threads}"
    )
    row = {
        "timestamp": timestamp,
        "run_id": run["run_id"],
        "game": run["game"],
        "env_id": run["env_id"],
        "node": run["node"],
        "owner_node": run.get("owner_node", run["node"]),
        "checkpoint_id": checkpoint_label,
        "checkpoint_path": model_path,
        "checkpoint": checkpoint_label,
        "checkpoint_episode": checkpoint_episode,
        "model": model_path,
        "episodes": str(args.episodes),
        "epsilon": str(args.epsilon),
        "frame_skip": str(args.frame_skip),
        "noop_max": str(args.noop_max),
        "max_steps_per_episode": str(args.max_steps_per_episode),
        "threads": str(args.threads),
        "mean": parsed.get("mean", ""),
        "std": parsed.get("std", ""),
        "min": parsed.get("min", ""),
        "max": parsed.get("max", ""),
        "host": host,
        "status": status,
        "log": log_path,
        "command": command,
        "requested_by": args.requested_by,
        "trigger": args.trigger,
        "milestone_step": str(args.milestone_step or ""),
    }
    if not args.dry_run:
        append_eval_result(runner, run, row)

    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    print(f"eval status: {status}")
    print(f"registry: {str(args.registry) if args.registry else canonical_registry_label()}")
    print(f"resource check: {resource_msg}")
    print(f"eval log: {log_path}")
    if status == "ok":
        print(f"mean={row['mean']} std={row['std']} min={row['min']} max={row['max']}")
    sys.exit(0 if status == "ok" or args.dry_run else proc.returncode or 1)


if __name__ == "__main__":
    main()
