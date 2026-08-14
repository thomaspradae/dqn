#!/usr/bin/env python3
import argparse
import csv
import io
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NODE_INFO = {
    "ofi1": {
        "instance": "100.107.98.78:9100",
        "ssh": "uace@100.107.98.78",
        "hostnames": {"uace-ofi-01", "ofi1"},
    },
    "ofi2": {
        "instance": "100.127.50.126:9100",
        "ssh": "uace@100.127.50.126",
        "hostnames": {"uace-ofi-02", "ofi2"},
    },
    "old1": {
        "instance": "100.80.3.43:9100",
        "ssh": "uace@100.80.3.43",
        "hostnames": {"uace-old-01", "old1"},
    },
    "old2": {
        "instance": "100.118.75.20:9100",
        "ssh": "uace@100.118.75.20",
        "hostnames": {"uace-old-02", "old2"},
    },
}

PROM_CANDIDATES = (
    "http://100.107.98.78:9090",
    "http://100.127.50.126:9090",
    "http://100.80.3.43:9090",
)

CANONICAL_REGISTRY_TARGET = os.environ.get("DQN_RESEARCH_REGISTRY_TARGET", "uace@100.107.98.78")
CANONICAL_REGISTRY_PATH = os.environ.get("DQN_RESEARCH_REGISTRY_PATH", "/home/uace/dqn/atari/run_registry.csv")
CANONICAL_QUEUE_PATH = os.environ.get("DQN_RESEARCH_QUEUE_PATH", "/home/uace/dqn/atari/eval_queue.csv")

REQUIRED_CSV_FIELDS = {
    "run_id",
    "algo",
    "game",
    "env_id",
    "node",
    "owner_node",
    "ssh_target",
    "type",
    "job_id",
    "run_dir",
    "log",
    "seed",
    "status",
    "target_steps",
    "step_unit",
    "checkpoint_every",
    "eval_every_steps",
    "project_dir",
    "venv_path",
    "eval_script",
    "checkpoint_pattern",
    "metric_type",
    "notes",
}


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


class ShellRunner:
    def __init__(self) -> None:
        self.local_hostname = socket.gethostname().split(".")[0]
        self.ssh_base = shlex.split(
            os.environ.get(
                "DQN_RESEARCH_SSH",
                "ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new",
            )
        )

    def is_local_node(self, node: str) -> bool:
        forced = os.environ.get("DQN_RESEARCH_LOCAL_NODE")
        if forced:
            return forced == node
        return self.local_hostname in NODE_INFO.get(node, {}).get("hostnames", set())

    def target_for(self, run: dict[str, str]) -> str:
        if run.get("ssh_target"):
            return run["ssh_target"]
        if run.get("node_target"):
            return run["node_target"]
        node = run.get("owner_node") or run.get("node", "")
        return NODE_INFO.get(node, {}).get("ssh", "")

    def run_shell(self, run: dict[str, str], command: str, timeout: int = 20) -> CommandResult:
        node = run.get("owner_node") or run.get("node", "")
        if self.is_local_node(node):
            args = ["bash", "-lc", command]
        else:
            target = self.target_for(run)
            if not target:
                return CommandResult("", f"no ssh target for node {node}", 2)
            args = self.ssh_base + [target, "bash -lc " + shlex.quote(command)]

        try:
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(exc.stdout or "", exc.stderr or "", 124, True)

    def read_file(self, run: dict[str, str], path: str, timeout: int = 25) -> str:
        qpath = shlex.quote(path)
        result = self.run_shell(run, f"test -r {qpath} && cat {qpath} || true", timeout)
        return result.stdout

    def tail_file(self, run: dict[str, str], path: str, lines: int = 120, timeout: int = 15) -> str:
        qpath = shlex.quote(path)
        result = self.run_shell(run, f"test -r {qpath} && tail -n {int(lines)} {qpath} || true", timeout)
        return result.stdout

    def stat_mtime(self, run: dict[str, str], path: str) -> int | None:
        qpath = shlex.quote(path)
        result = self.run_shell(run, f"stat -c %Y {qpath} 2>/dev/null || true", 10)
        text = result.stdout.strip()
        return int(text) if text.isdigit() else None


def default_registry_path() -> Path:
    return Path(__file__).resolve().with_name("run_registry.csv")


def canonical_registry_label() -> str:
    return f"{CANONICAL_REGISTRY_TARGET}:{CANONICAL_REGISTRY_PATH}"


def normalize_run(row: dict[str, str]) -> dict[str, str]:
    cleaned = {k: (v or "").strip() for k, v in row.items()}
    cleaned.setdefault("algo", cleaned.get("type", ""))
    cleaned.setdefault("owner_node", cleaned.get("node", ""))
    cleaned.setdefault("ssh_target", cleaned.get("node_target", ""))
    cleaned.setdefault("venv_path", cleaned.get("venv", ""))
    cleaned.setdefault("eval_script", "eval.py" if cleaned.get("type") == "scratch_dqn" else "")
    cleaned.setdefault(
        "checkpoint_pattern",
        "q_net_ep{episode}.pt" if cleaned.get("type") == "scratch_dqn" else "*.zip",
    )
    cleaned.setdefault(
        "metric_type",
        "scratch_rewards" if cleaned.get("type") == "scratch_dqn" else "sb3_monitor",
    )
    if not cleaned.get("ssh_target"):
        node = cleaned.get("owner_node") or cleaned.get("node", "")
        cleaned["ssh_target"] = NODE_INFO.get(node, {}).get("ssh", "")
    if not cleaned.get("node") and cleaned.get("owner_node"):
        cleaned["node"] = cleaned["owner_node"]
    if not cleaned.get("owner_node") and cleaned.get("node"):
        cleaned["owner_node"] = cleaned["node"]
    if not cleaned.get("algo"):
        cleaned["algo"] = cleaned.get("type", "")
    return cleaned


def parse_registry_text(text: str, source: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = set(reader.fieldnames or [])
    if "node_target" in fieldnames or "venv" in fieldnames:
        required = {"run_id", "game", "env_id", "node", "type", "run_dir", "log"}
    else:
        required = REQUIRED_CSV_FIELDS
    missing = required.difference(fieldnames)
    if missing:
        raise SystemExit(f"registry {source} missing columns: {', '.join(sorted(missing))}")
    rows = []
    for row in reader:
        cleaned = normalize_run({k: (v or "") for k, v in row.items()})
        if cleaned.get("run_id"):
            rows.append(cleaned)
    return rows


def read_canonical_registry() -> tuple[str, str] | None:
    runner = ShellRunner()
    local_path = Path(CANONICAL_REGISTRY_PATH)
    if runner.is_local_node("ofi1") and local_path.exists():
        return local_path.read_text(), str(local_path)

    args = runner.ssh_base + [
        CANONICAL_REGISTRY_TARGET,
        "cat " + shlex.quote(CANONICAL_REGISTRY_PATH),
    ]
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout, canonical_registry_label()
    return None


def load_registry(path: Path | None = None) -> list[dict[str, str]]:
    if path is not None:
        return parse_registry_text(path.read_text(), str(path))

    canonical = read_canonical_registry()
    if canonical is not None:
        text, source = canonical
        return parse_registry_text(text, source)

    fallback = default_registry_path()
    return parse_registry_text(fallback.read_text(), str(fallback))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def local_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_int(value: str | int | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(str(value).replace(",", "").strip())
    except ValueError:
        return default


def safe_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        v = float(str(value).strip())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def fmt_float(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def fmt_signed_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.0f}%"


def fmt_steps(value: int | None) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def fmt_episode(value: int | str | None) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"ep{int(str(value))}"
    except ValueError:
        return str(value)


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 90:
        return f"{int(seconds)}s ago"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m ago"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f}h ago"
    return f"{hours / 24:.1f}d ago"


def percent_done(done: int | None, target: int) -> float | None:
    if done is None or target <= 0:
        return None
    return min(999.9, 100.0 * done / target)


def training_drop_percent(training: dict[str, Any]) -> float | None:
    last100 = safe_float(training.get("last100"))
    best100 = safe_float(training.get("best_last100"))
    if last100 is None or best100 is None or best100 == 0:
        return None
    return ((last100 - best100) / abs(best100)) * 100.0


def likely_training_state(training: dict[str, Any]) -> str:
    drop = training_drop_percent(training)
    if drop is None:
        return "unknown"
    if drop <= -60:
        return "likely unstable"
    if drop <= -30:
        return "watch degradation"
    if drop >= -15:
        return "near best"
    return "below best"


def run_display_name(run: dict[str, str]) -> str:
    run_id = run.get("run_id", "")
    game = run.get("game", run_id)
    if run.get("type") == "sb3":
        return f"SB3 {game}"
    match = re.match(r".*_v2_seed(\d+)$", run_id)
    if match:
        return f"{game} v2 seed{match.group(1)}"
    if run_id.endswith("_rep2"):
        return f"{game} rep2"
    if run_id.endswith("_rep1"):
        return f"{game} rep1"
    return f"{game} {run_id}".strip()


def run_status(run: dict[str, str]) -> str:
    return (run.get("status") or "").strip().lower()


def is_archived_run(run: dict[str, str]) -> bool:
    return run_status(run) in {"stopped", "archived", "finished"}


def is_v2_run(run: dict[str, str]) -> bool:
    return "_v2_" in run.get("run_id", "") or "v2" in run.get("notes", "").lower()


def first_eval_step(run: dict[str, str]) -> int:
    return safe_int(run.get("eval_every_steps"), 5_000_000) or 5_000_000


def run_phase(run: dict[str, str], current_step: int | None) -> str:
    if current_step is None:
        return "waiting for first log"
    if not is_v2_run(run):
        return "running"
    if current_step < 50_000:
        return "initializing"
    if current_step < 1_000_000:
        return "early training / replay filling"
    if current_step < first_eval_step(run):
        return "training, no eval due"
    return "milestone evals active"


def next_eval_text(run: dict[str, str], current_step: int | None) -> str:
    milestone = first_eval_step(run)
    if current_step is None or current_step < milestone:
        return f"Next eval: {fmt_steps(milestone)} milestone"
    return "Next eval: milestone watcher active"


def choose_prom() -> str:
    env_prom = os.environ.get("POORMANS_PROM")
    candidates = [env_prom] if env_prom else list(PROM_CANDIDATES)
    for prom in candidates:
        try:
            with urllib.request.urlopen(f"{prom}/-/healthy", timeout=2) as resp:
                if resp.status < 400:
                    return prom
        except Exception:
            continue
    return candidates[0]


def prom_query(prom: str, query: str) -> float | None:
    url = prom + "/api/v1/query?query=" + urllib.parse.quote(query)
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            data = json.loads(resp.read())
        result = data.get("data", {}).get("result", [])
        if not result:
            return None
        return safe_float(result[0]["value"][1])
    except Exception:
        return None


def load_cluster_resources(prom: str) -> dict[str, dict[str, float | None]]:
    resources: dict[str, dict[str, float | None]] = {}
    for label, info in NODE_INFO.items():
        inst = info["instance"]
        resources[label] = {
            "cpu": prom_query(
                prom,
                f'100-(avg by(instance)(rate(node_cpu_seconds_total{{mode="idle",instance="{inst}"}}[2m]))*100)',
            ),
            "ram": prom_query(
                prom,
                f'100-((node_memory_MemAvailable_bytes{{instance="{inst}"}}/node_memory_MemTotal_bytes{{instance="{inst}"}})*100)',
            ),
            "load": prom_query(prom, f'node_load1{{instance="{inst}"}}'),
        }
    return resources


def read_eval_queue_rows() -> list[dict[str, str]]:
    runner = ShellRunner()
    queue_node = {
        "run_id": "__queue__",
        "node": "ofi1",
        "owner_node": "ofi1",
        "ssh_target": CANONICAL_REGISTRY_TARGET,
    }
    text = runner.read_file(queue_node, CANONICAL_QUEUE_PATH, 12)
    if not text.strip():
        return []
    try:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(io.StringIO(text))]
    except csv.Error:
        return [{"status": "failed", "last_error": "invalid eval_queue.csv"}]


def eval_queue_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "total": 0}
    for row in rows:
        status = (row.get("status") or "").strip() or "pending"
        counts[status] = counts.get(status, 0) + 1
        counts["total"] += 1
    return counts


def actual_eval_process_counts(node_labels: set[str] | None = None) -> dict[str, int]:
    runner = ShellRunner()
    command = (
        "ps -eo comm=,args= | "
        "awk 'tolower($1) ~ /python/ && $0 ~ /(eval.py|eval_checkpoint.py)/ "
        "&& $0 !~ /auto_eval_milestones.py/ {print}'"
    )
    counts: dict[str, int] = {}
    labels = node_labels or set(NODE_INFO)
    for label in sorted(labels):
        info = NODE_INFO.get(label)
        if info is None:
            continue
        node = {
            "run_id": f"__eval_processes_{label}__",
            "node": label,
            "owner_node": label,
            "ssh_target": info.get("ssh", ""),
        }
        result = runner.run_shell(node, command, 5)
        if result.returncode != 0 and not result.stdout.strip():
            counts[label] = 0
            continue
        counts[label] = len([line for line in result.stdout.splitlines() if line.strip()])
    return counts


def read_live_slurm_jobs() -> dict[str, dict[str, str]]:
    runner = ShellRunner()
    slurm_node = {
        "run_id": "__slurm__",
        "node": "ofi1",
        "owner_node": "ofi1",
        "ssh_target": CANONICAL_REGISTRY_TARGET,
    }
    result = runner.run_shell(
        slurm_node,
        'command -v squeue >/dev/null 2>&1 && squeue -h -o "%i|%j|%T|%M|%N" || true',
        12,
    )
    jobs: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        job_id, name, state, elapsed, nodelist = [part.strip() for part in parts]
        if job_id:
            jobs[job_id] = {
                "job_id": job_id,
                "name": name,
                "state": state,
                "elapsed": elapsed,
                "nodelist": nodelist,
            }
    return jobs


def live_job_node(job: dict[str, str], runs_by_job: dict[str, dict[str, str]]) -> str | None:
    run = runs_by_job.get(job.get("job_id", ""))
    if run is not None:
        return run.get("owner_node") or run.get("node")
    nodelist = job.get("nodelist", "")
    for label, info in NODE_INFO.items():
        if nodelist == label or any(host in nodelist for host in info.get("hostnames", set())):
            return label
    return None


def parse_rewards(text: str, window: int = 100) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    if not text.strip():
        return {"episodes": 0}
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            episode = safe_int(row.get("episode"), -1)
            reward = safe_float(row.get("reward"))
            epsilon = safe_float(row.get("epsilon"))
            if episode >= 0 and reward is not None:
                rows.append({"episode": episode, "reward": reward, "epsilon": epsilon})
    except csv.Error:
        return {"episodes": 0, "error": "invalid rewards.csv"}

    if not rows:
        return {"episodes": 0}

    latest = rows[-1]
    last_window = rows[-window:]
    last_avg = sum(float(r["reward"]) for r in last_window) / len(last_window)

    best_avg = None
    best_episode = None
    if len(rows) >= window:
        rolling_sum = sum(float(r["reward"]) for r in rows[:window])
        best_avg = rolling_sum / window
        best_episode = int(rows[window - 1]["episode"])
        for idx in range(window, len(rows)):
            rolling_sum += float(rows[idx]["reward"]) - float(rows[idx - window]["reward"])
            avg = rolling_sum / window
            if best_avg is None or avg > best_avg:
                best_avg = avg
                best_episode = int(rows[idx]["episode"])

    return {
        "episodes": len(rows),
        "latest_episode": int(latest["episode"]),
        "latest_reward": float(latest["reward"]),
        "latest_epsilon": latest["epsilon"],
        "last100": last_avg,
        "best_last100": best_avg,
        "best_last100_episode": best_episode,
    }


def parse_log_progress(text: str) -> dict[str, int | None]:
    env_steps = [int(x) for x in re.findall(r"\benv_step=(\d+)", text)]
    agent_steps = [int(x) for x in re.findall(r"\bagent_step=(\d+)", text)]
    train_steps = [int(x) for x in re.findall(r"\btrain_step=(\d+)", text)]
    episodes = [int(x) for x in re.findall(r"\bepisode=(\d+)", text)]
    return {
        "env_step": env_steps[-1] if env_steps else None,
        "agent_step": agent_steps[-1] if agent_steps else None,
        "train_step": train_steps[-1] if train_steps else None,
        "episode": episodes[-1] if episodes else None,
    }


def parse_q_checkpoints(listing: str) -> dict[str, Any]:
    checkpoints = []
    for line in listing.splitlines():
        name = Path(line.strip()).name
        match = re.match(r"q_net_ep(\d+)\.pt$", name)
        if match:
            checkpoints.append(int(match.group(1)))
    latest = max(checkpoints) if checkpoints else None
    return {"episodes": sorted(checkpoints), "latest_episode": latest}


def parse_checkpoint_manifest(text: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    rows = []
    try:
        for index, row in enumerate(csv.DictReader(io.StringIO(text))):
            episode = safe_int(row.get("episode"), -1)
            env_step = safe_int(row.get("env_step"), -1)
            agent_step = safe_int(row.get("agent_step"), -1)
            if episode < 0:
                continue
            cleaned: dict[str, Any] = {k: (v or "").strip() for k, v in row.items()}
            cleaned["_row_index"] = index
            cleaned["_episode"] = episode
            cleaned["_env_step"] = env_step if env_step >= 0 else None
            cleaned["_agent_step"] = agent_step if agent_step >= 0 else None
            rows.append(cleaned)
    except csv.Error:
        return []
    return rows


def select_active_checkpoint_episode(
    checkpoints: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    current_episode: int | None,
) -> tuple[int | None, dict[str, Any] | None, list[int]]:
    file_episodes = checkpoints.get("episodes", [])
    stale_episodes = [ep for ep in file_episodes if current_episode is not None and ep > current_episode]

    candidates = manifest_rows
    if current_episode is not None:
        candidates = [row for row in manifest_rows if row.get("_episode", -1) <= current_episode]
    if candidates:
        row = max(candidates, key=lambda item: (str(item.get("timestamp", "")), item.get("_row_index", 0)))
        return int(row["_episode"]), row, stale_episodes

    valid_files = file_episodes
    if current_episode is not None:
        valid_files = [ep for ep in file_episodes if ep <= current_episode]
    if valid_files:
        return max(valid_files), None, stale_episodes
    return None, None, stale_episodes


def checkpoint_glob(run: dict[str, str]) -> str:
    pattern = run.get("checkpoint_pattern") or "q_net_ep{episode}.pt"
    if "{episode}" in pattern:
        return pattern.replace("{episode}", "*")
    if "{step}" in pattern:
        return pattern.replace("{step}", "*")
    return pattern


def checkpoint_path_for_episode(run: dict[str, str], episode: int | None) -> str | None:
    if episode is None:
        return None
    pattern = run.get("checkpoint_pattern") or "q_net_ep{episode}.pt"
    if "{episode}" in pattern:
        name = pattern.format(episode=episode)
    else:
        name = f"q_net_ep{episode}.pt"
    return f"{run['run_dir']}/{name}"


def nearest_checkpoint_episode(episodes: list[int], target_episode: int | None) -> int | None:
    if target_episode is None or not episodes:
        return None
    return min(episodes, key=lambda ep: (abs(ep - target_episode), ep > target_episode, ep))


def parse_eval_results(text: str) -> dict[str, Any]:
    if not text.strip():
        return {"count": 0}
    rows = []
    try:
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("status", "ok") != "ok":
                continue
            mean = safe_float(row.get("mean"))
            if mean is None:
                continue
            row["_mean"] = mean
            row["_timestamp"] = row.get("timestamp", "")
            rows.append(row)
    except csv.Error:
        return {"count": 0, "error": "invalid eval_results.csv"}
    if not rows:
        return {"count": 0}
    latest = rows[-1]
    best = max(rows, key=lambda r: float(r["_mean"]))
    return {"count": len(rows), "latest": latest, "best": best, "rows": rows}


def eval_for_checkpoint(evals: dict[str, Any], episode: int | None) -> dict[str, Any] | None:
    if episode is None or evals.get("count", 0) == 0:
        return None
    wanted_id = f"ep{episode}"
    wanted_episode = str(episode)
    matches = []
    for row in evals.get("rows", []):
        if (
            row.get("checkpoint_id") == wanted_id
            or row.get("checkpoint") == wanted_id
            or row.get("checkpoint_episode") == wanted_episode
        ):
            matches.append(row)
    return matches[-1] if matches else None


def newest_checkpoint_eval(evals: dict[str, Any]) -> dict[str, Any] | None:
    rows = []
    for row in evals.get("rows", []):
        episode = safe_int(row.get("checkpoint_episode"), -1)
        if episode >= 0:
            rows.append((episode, row.get("_timestamp", ""), row))
    if not rows:
        return None
    return max(rows, key=lambda item: (item[0], item[1]))[2]


def parse_sb3_log(text: str) -> dict[str, Any]:
    fields = {
        "ep_len_mean": r"\|\s+ep_len_mean\s+\|\s+([0-9.]+)",
        "ep_rew_mean": r"\|\s+ep_rew_mean\s+\|\s+([-0-9.]+)",
        "exploration_rate": r"\|\s+exploration_rate\s+\|\s+([0-9.]+)",
        "episodes": r"\|\s+episodes\s+\|\s+([0-9]+)",
        "fps": r"\|\s+fps\s+\|\s+([0-9]+)",
        "time_elapsed": r"\|\s+time_elapsed\s+\|\s+([0-9]+)",
        "total_timesteps": r"\|\s+total_timesteps\s+\|\s+([0-9]+)",
        "loss": r"\|\s+loss\s+\|\s+([-0-9.eE]+)",
        "n_updates": r"\|\s+n_updates\s+\|\s+([0-9]+)",
    }
    parsed: dict[str, Any] = {}
    for key, pattern in fields.items():
        matches = re.findall(pattern, text)
        if not matches:
            continue
        value = matches[-1]
        parsed[key] = safe_float(value) if "." in value or "e" in value.lower() else safe_int(value)
    return parsed


def parse_sb3_monitor(text: str, window: int = 100) -> dict[str, Any]:
    if not text.strip():
        return {}
    payload = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    if not payload.strip():
        return {}
    try:
        rows = list(csv.DictReader(io.StringIO(payload)))
    except csv.Error:
        return {}
    rewards = [safe_float(row.get("r")) for row in rows]
    rewards = [r for r in rewards if r is not None]
    if not rewards:
        return {}
    last_window = rewards[-window:]
    return {
        "monitor_episodes": len(rewards),
        "monitor_latest_reward": rewards[-1],
        "monitor_last100": sum(last_window) / len(last_window),
    }


def parse_sb3_checkpoints(listing: str) -> dict[str, Any]:
    steps = []
    final_exists = False
    for line in listing.splitlines():
        name = Path(line.strip()).name
        match = re.search(r"_(\d+)_steps\.zip$", name)
        if match:
            steps.append(int(match.group(1)))
        if name == "final_model.zip":
            final_exists = True
    return {
        "steps": sorted(steps),
        "latest_steps": max(steps) if steps else None,
        "final_exists": final_exists,
    }


def get_job_state(runner: ShellRunner, run: dict[str, str]) -> dict[str, str]:
    if is_archived_run(run):
        return {"state": run_status(run)}
    job_id = run.get("job_id", "")
    if not job_id or job_id == "none":
        return {}
    qjob = shlex.quote(job_id)
    result = runner.run_shell(
        run,
        f'command -v squeue >/dev/null 2>&1 && squeue --noheader -j {qjob} -o "%T|%M|%R" || true',
        12,
    )
    line = result.stdout.strip().splitlines()
    if not line:
        return {"state": "not in squeue"}
    parts = line[0].split("|", 2)
    while len(parts) < 3:
        parts.append("")
    return {"state": parts[0], "elapsed": parts[1], "reason": parts[2]}


def build_scratch_snapshot(runner: ShellRunner, run: dict[str, str]) -> dict[str, Any]:
    run_dir = run["run_dir"]
    log_path = run["log"]
    rewards = parse_rewards(runner.read_file(run, f"{run_dir}/rewards.csv", 30))
    log_tail = runner.tail_file(run, log_path, 600, 20)
    log_progress = parse_log_progress(log_tail)
    ckpt_glob = checkpoint_glob(run)
    checkpoint_listing = runner.run_shell(
        run,
        f"find {shlex.quote(run_dir)} -maxdepth 1 -type f -name {shlex.quote(ckpt_glob)} -print 2>/dev/null || true",
        20,
    ).stdout
    checkpoints = parse_q_checkpoints(checkpoint_listing)
    checkpoint_manifest = parse_checkpoint_manifest(runner.read_file(run, f"{run_dir}/checkpoints.csv", 20))
    evals = parse_eval_results(runner.read_file(run, f"{run_dir}/eval_results.csv", 20))
    mtime = runner.stat_mtime(run, log_path)
    age = time.time() - mtime if mtime is not None else None
    target = safe_int(run.get("target_steps"))
    env_step = log_progress.get("env_step")
    agent_step = log_progress.get("agent_step")
    current_episode = log_progress.get("episode") or rewards.get("latest_episode")
    latest_ckpt, latest_manifest_row, stale_episodes = select_active_checkpoint_episode(
        checkpoints,
        checkpoint_manifest,
        current_episode,
    )
    checkpoints["latest_episode_raw"] = checkpoints.get("latest_episode")
    checkpoints["latest_episode"] = latest_ckpt
    checkpoints["latest_manifest_row"] = latest_manifest_row
    checkpoints["stale_episodes"] = stale_episodes
    if env_step is None and latest_manifest_row is not None:
        env_step = latest_manifest_row.get("_env_step")
    if agent_step is None and latest_manifest_row is not None:
        agent_step = latest_manifest_row.get("_agent_step")
    current_step = agent_step if run.get("step_unit") == "decisions" else env_step
    evals["latest_checkpoint_eval"] = eval_for_checkpoint(evals, latest_ckpt)
    evals["newest_checkpoint_eval"] = newest_checkpoint_eval(evals)

    return {
        "run": run,
        "job": get_job_state(runner, run),
        "training": rewards,
        "log_progress": log_progress,
        "env_step": env_step,
        "agent_step": agent_step,
        "current_step": current_step,
        "percent_done": percent_done(current_step, target),
        "checkpoints": checkpoints,
        "latest_checkpoint_path": checkpoint_path_for_episode(run, latest_ckpt),
        "evals": evals,
        "log_age_seconds": age,
        "verdict": scratch_verdict(run, evals, rewards, current_step, percent_done(current_step, target), age),
    }


def build_sb3_snapshot(runner: ShellRunner, run: dict[str, str]) -> dict[str, Any]:
    run_dir = run["run_dir"]
    log_path = run["log"]
    log_tail = runner.tail_file(run, log_path, 1000, 20)
    sb3_log = parse_sb3_log(log_tail)
    monitor = parse_sb3_monitor(runner.read_file(run, f"{run_dir}/monitor.csv", 25))
    listing = runner.run_shell(
        run,
        f"find {shlex.quote(run_dir)} -maxdepth 1 -type f -name '*.zip' -print 2>/dev/null || true",
        20,
    ).stdout
    checkpoints = parse_sb3_checkpoints(listing)
    mtime = runner.stat_mtime(run, log_path)
    age = time.time() - mtime if mtime is not None else None
    target = safe_int(run.get("target_steps"))
    current = safe_int(str(sb3_log.get("total_timesteps", "")), None) if sb3_log else None
    if current == 0:
        current = None

    return {
        "run": run,
        "job": {},
        "sb3": {**monitor, **sb3_log},
        "current_step": current,
        "percent_done": percent_done(current, target),
        "checkpoints": checkpoints,
        "evals": parse_eval_results(runner.read_file(run, f"{run_dir}/eval_results.csv", 20)),
        "log_age_seconds": age,
        "verdict": sb3_verdict(run, current, target, age, checkpoints),
    }


def build_snapshot(runner: ShellRunner, run: dict[str, str]) -> dict[str, Any]:
    if run.get("type") == "sb3":
        return build_sb3_snapshot(runner, run)
    return build_scratch_snapshot(runner, run)


def scratch_verdict(
    run: dict[str, str],
    evals: dict[str, Any],
    training: dict[str, Any],
    current_step: int | None,
    pct: float | None,
    log_age: float | None,
) -> str:
    if run_status(run) == "archived":
        return "archived_uncontrolled" if "uncontrolled" in run.get("notes", "").lower() else "archived"
    if run_status(run) == "stopped":
        return "stopped_diagnostic" if "diagnostic" in run.get("notes", "").lower() else "stopped"
    if training.get("error") or evals.get("error"):
        return "error"
    if run.get("status") == "finished" or (pct is not None and pct >= 100):
        return "finished" if evals.get("count", 0) else "needs_eval"
    if log_age is not None and log_age > 1800 and run.get("status") == "running":
        return "error"
    if is_v2_run(run) and current_step is not None and current_step < first_eval_step(run):
        return "running_pre_eval"
    if evals.get("count", 0) == 0:
        return "needs_eval"

    latest = evals.get("latest_checkpoint_eval") or evals.get("newest_checkpoint_eval") or evals.get("latest", {})
    best = evals.get("best", {})
    latest_mean = safe_float(latest.get("mean"))
    best_mean = safe_float(best.get("mean"))
    if latest_mean is None:
        return "error"
    if best_mean is not None and best_mean < 20:
        return "stable_but_weak"
    if best_mean is not None and best_mean > 0 and latest_mean < 0.5 * best_mean and evals.get("count", 0) >= 2:
        return "unstable"
    if latest_mean <= 0 and pct is not None and pct >= 50 and evals.get("count", 0) >= 2:
        return "collapsed_candidate"
    if latest_mean <= 0:
        return "weak"

    last100 = safe_float(training.get("last100"))
    best_last100 = safe_float(training.get("best_last100"))
    if best_last100 is not None and last100 is not None and last100 >= 0.8 * best_last100:
        return "learning"
    return "healthy"


def sb3_verdict(
    run: dict[str, str],
    current: int | None,
    target: int,
    age: float | None,
    checkpoints: dict[str, Any],
) -> str:
    if run_status(run) == "archived":
        return "archived"
    if run_status(run) == "stopped":
        return "stopped"
    if run.get("status") == "finished":
        return "finished"
    if checkpoints.get("final_exists") or (current is not None and target and current >= target):
        return "finished"
    if age is not None and age > 1800:
        return "error"
    if current is not None:
        return "healthy"
    return "needs_eval"


def render_resources(
    resources: dict[str, dict[str, float | None]],
    runs: list[dict[str, str]],
    live_jobs: dict[str, dict[str, str]] | None = None,
    compact: bool = False,
) -> str:
    jobs_by_node: dict[str, list[str]] = {label: [] for label in NODE_INFO}
    live_jobs = live_jobs or {}
    runs_by_job = {run.get("job_id", ""): run for run in runs if run.get("job_id") and run.get("job_id") != "none"}
    for job_id, job in live_jobs.items():
        label = live_job_node(job, runs_by_job)
        if not label or label not in jobs_by_node:
            continue
        run = runs_by_job.get(job_id)
        if run is not None:
            name = run_display_name(run)
        else:
            name = job.get("name") or f"job {job_id}"
        jobs_by_node[label].append(f"{name} job {job_id}".strip())

    lines = ["Cluster"]
    for label in NODE_INFO:
        row = resources.get(label, {})
        jobs = ", ".join(jobs_by_node.get(label, [])) or "idle"
        has_live_job = jobs != "idle"
        cpu = row.get("cpu")
        ram = row.get("ram")
        if cpu is None or ram is None:
            state = "unknown"
        elif cpu >= 98 or ram >= 98:
            state = "blocked"
        elif cpu >= 90 or ram >= 90:
            state = "busy"
        else:
            state = "available"
        suspect = ""
        if has_live_job and state == "available":
            state = "busy"
            if cpu is not None and cpu < 10:
                suspect = " (metrics suspect)"
        load = "" if compact else f", load {fmt_float(row.get('load'), 2)}"
        lines.append(
            f"{label}: {state}, {fmt_percent(row.get('cpu'))} CPU / {fmt_percent(row.get('ram'))} RAM"
            f"{load}{suspect}, {jobs}"
        )
    return "\n".join(lines)


def render_eval_capacity(
    resources: dict[str, dict[str, float | None]],
    queue_counts: dict[str, int],
    eval_process_counts: dict[str, int],
) -> str:
    blocked = 0
    hot = 0
    available = 0
    for row in resources.values():
        cpu = row.get("cpu")
        ram = row.get("ram")
        if cpu is None or ram is None:
            continue
        if cpu >= 98 or ram >= 98:
            blocked += 1
        elif cpu >= 90 or ram >= 90:
            hot += 1
        else:
            available += 1

    if available:
        state = "available"
    elif hot and not blocked:
        state = "hot, low-priority eval OK"
    elif hot:
        state = "mixed, queue or force"
    else:
        state = "blocked, queue only"

    pending = queue_counts.get("pending", 0)
    queue_running = queue_counts.get("running", 0)
    running_processes = sum(eval_process_counts.values())
    stale_running = max(0, queue_running - running_processes)
    return (
        f"Eval capacity: {state} | queue pending {pending}, queue_running {queue_running}, "
        f"running_processes {running_processes}, stale_running {stale_running}"
    )


def render_runs(runs: list[dict[str, str]]) -> str:
    lines = ["Registered runs:"]
    for run in runs:
        job = run.get("job_id") or "none"
        lines.append(
            f"{run['run_id']}: {run['game']} {run.get('algo', run['type'])} {run['status']} "
            f"owner={run.get('owner_node', run['node'])} job={job} seed={run['seed']}"
        )
    return "\n".join(lines)


def run_has_active_queue(queue_rows: list[dict[str, str]], run_id: str) -> bool:
    return any(
        row.get("run_id") == run_id and row.get("status") in {"pending", "running"}
        for row in queue_rows
    )


def visible_status_snapshots(
    snapshots: list[dict[str, Any]],
    queue_rows: list[dict[str, str]],
    live_jobs: dict[str, dict[str, str]],
    show_all: bool,
) -> list[dict[str, Any]]:
    if show_all:
        return snapshots
    visible = []
    live_job_ids = set(live_jobs)
    for snapshot in snapshots:
        run = snapshot["run"]
        if is_archived_run(run):
            continue
        if run.get("job_id") in live_job_ids:
            visible.append(snapshot)
            continue
        if run_has_active_queue(queue_rows, run["run_id"]):
            visible.append(snapshot)
            continue
        if snapshot.get("verdict") == "error":
            visible.append(snapshot)
    return visible


def render_status(
    snapshots: list[dict[str, Any]],
    resources: dict[str, dict[str, float | None]],
    registry_source: str,
    show_all: bool = False,
) -> str:
    live_jobs = read_live_slurm_jobs()
    runs = [s["run"] for s in snapshots]
    queue_rows = read_eval_queue_rows()
    queue_counts = eval_queue_counts(queue_rows)
    eval_nodes = {
        snapshot["run"].get("owner_node") or snapshot["run"].get("node", "")
        for snapshot in snapshots
        if snapshot["run"].get("type") == "scratch_dqn"
    }
    eval_nodes.add("ofi1")
    eval_nodes.discard("")
    eval_process_counts = actual_eval_process_counts(eval_nodes)
    visible_snapshots = visible_status_snapshots(snapshots, queue_rows, live_jobs, show_all)
    lines = [
        "DQN research status · briefing" + (" · all" if show_all else ""),
        local_timestamp(),
        "",
        render_resources(resources, runs, live_jobs=live_jobs, compact=True),
        render_eval_capacity(resources, queue_counts, eval_process_counts),
        "",
    ]
    if not visible_snapshots:
        lines.append("No active registered runs.")
        lines.append("")
    for snapshot in visible_snapshots:
        lines.extend(render_run_block(snapshot))
        lines.append("")
    queued = render_queued_evals(queue_rows)
    if queued:
        lines.extend(queued)
        lines.append("")
    return "\n".join(lines).rstrip()


def render_run_block(snapshot: dict[str, Any]) -> list[str]:
    run = snapshot["run"]
    header = run_display_name(run)
    if run_status(run) in {"stopped", "archived"}:
        label = "stopped diagnostic run" if run_status(run) == "stopped" else "archived diagnostic run"
        header = f"{header} · {label}"
    elif run.get("job_id") and run["job_id"] != "none":
        job = snapshot.get("job", {})
        state = job.get("state", run.get("status", "")).lower()
        header = f"{header} · job {run['job_id']} {state}".rstrip()

    lines = [header]
    target = safe_int(run.get("target_steps"))
    unit = run.get("step_unit", "steps")
    current = snapshot.get("current_step")
    pct = snapshot.get("percent_done")
    if current is not None:
        lines.append(f"Progress: {fmt_steps(current)} / {fmt_steps(target)} {unit}, {fmt_percent(pct)}")
    else:
        lines.append(f"Progress: unknown / {fmt_steps(target)} {unit}")
    phase = run_phase(run, current)
    if is_v2_run(run) or phase != "running":
        lines.append(f"Phase: {phase}")

    if run.get("type") == "sb3":
        sb3 = snapshot.get("sb3", {})
        lines.append(
            f"Seed: {run.get('seed', 'N/A')} | FPS: {sb3.get('fps', 'N/A')} | "
            f"Mean episode reward: {fmt_float(safe_float(str(sb3.get('ep_rew_mean'))) if sb3.get('ep_rew_mean') is not None else None, 2)}"
        )
        ckpt = snapshot.get("checkpoints", {})
        ckpt_text = fmt_steps(ckpt.get("latest_steps")) + " steps" if ckpt.get("latest_steps") else "none"
        if ckpt.get("final_exists"):
            ckpt_text += ", final"
        lines.append(f"Latest ckpt: {ckpt_text}")
    else:
        training = snapshot.get("training", {})
        latest_ep = snapshot.get("checkpoints", {}).get("latest_episode")
        ckpt_eps = snapshot.get("checkpoints", {}).get("episodes", [])
        best_candidate = nearest_checkpoint_episode(ckpt_eps, training.get("best_last100_episode"))
        drop = training_drop_percent(training)
        lines.append(f"Latest ckpt: ep{latest_ep}" if latest_ep else "Latest ckpt: none")
        lines.append(
            f"Training: last100 {fmt_float(training.get('last100'), 1)}, "
            f"best {fmt_float(training.get('best_last100'), 1)} @ {fmt_episode(training.get('best_last100_episode'))}"
        )
        lines.append(f"Drop from best: {fmt_signed_percent(drop)}")
        lines.append(
            f"Best checkpoint candidate: ep{best_candidate}"
            if best_candidate is not None
            else "Best checkpoint candidate: unknown"
        )
        stale = snapshot.get("checkpoints", {}).get("stale_episodes", [])
        if stale:
            lines.append(f"Checkpoint note: ignoring {len(stale)} stale ckpts up to ep{max(stale)}")

    lines.append(render_eval_line(snapshot))
    lines.append(render_next_evidence(snapshot))
    lines.append(f"Verdict: {render_verdict_line(snapshot)}")
    lines.append(f"Log fresh: {fmt_age(snapshot.get('log_age_seconds'))}")
    return lines


def render_next_evidence(snapshot: dict[str, Any]) -> str:
    run = snapshot["run"]
    if run.get("type") != "scratch_dqn":
        return "Next evidence: monitor until checkpoint/final eval"
    current = snapshot.get("current_step")
    if is_v2_run(run) and (current is None or current < first_eval_step(run)):
        return next_eval_text(run, current)
    evals = snapshot.get("evals", {})
    if evals.get("count", 0) == 0:
        return "Next evidence: eval best + latest"
    if evals.get("latest_checkpoint_eval") is None:
        return "Next evidence: eval current latest + compare against best_eval"
    latest = evals.get("latest_checkpoint_eval") or evals.get("newest_checkpoint_eval") or evals.get("latest", {})
    best = evals.get("best", {})
    if latest == best:
        return "Next evidence: eval latest if training moved since best_eval"
    return "Next evidence: compare latest vs best_eval"


def render_verdict_line(snapshot: dict[str, Any]) -> str:
    verdict = snapshot.get("verdict", "unknown")
    run = snapshot["run"]
    if verdict == "running_pre_eval":
        phase = run_phase(run, snapshot.get("current_step"))
        return "running, pre-learning" if phase in {"initializing", "early training / replay filling"} else "running"
    if run.get("type") == "scratch_dqn" and verdict == "needs_eval":
        state = likely_training_state(snapshot.get("training", {}))
        if state != "unknown":
            return f"needs_eval, {state}"
    if verdict == "stable_but_weak":
        return "stable but weak"
    return str(verdict)


def short_run_id(run_id: str) -> str:
    return run_id.split("_", 1)[0]


def selector_matches(left: str, right: str) -> bool:
    aliases = {
        "best": {"best", "best_train"},
        "best_train": {"best", "best_train"},
        "latest": {"latest"},
    }
    return right in aliases.get(left, {left})


def has_active_queue_request(queue_rows: list[dict[str, str]], run_id: str, selector: str) -> bool:
    for row in queue_rows:
        if row.get("run_id") != run_id:
            continue
        if row.get("status") not in {"pending", "running"}:
            continue
        if selector_matches(selector, row.get("checkpoint_selector", "")):
            return True
    return False


def render_queued_evals(queue_rows: list[dict[str, str]]) -> list[str]:
    active = [row for row in queue_rows if row.get("status") in {"pending", "running"}]
    if not active:
        return []
    lines = ["Already queued evals"]
    for row in active:
        trigger = row.get("trigger") or "manual"
        lines.append(
            f"{row.get('status', 'pending')} {short_run_id(row.get('run_id', ''))} "
            f"{row.get('checkpoint_selector', 'latest')} {row.get('episodes', '10')}ep ({trigger})"
        )
    return lines


def render_eval_suggestions(snapshots: list[dict[str, Any]], queue_rows: list[dict[str, str]]) -> list[str]:
    suggestions = []
    for snapshot in snapshots:
        run = snapshot["run"]
        if run.get("type") != "scratch_dqn":
            continue
        evals = snapshot.get("evals", {})
        checkpoints = snapshot.get("checkpoints", {})
        latest_ep = checkpoints.get("latest_episode")
        best_candidate = nearest_checkpoint_episode(
            checkpoints.get("episodes", []),
            snapshot.get("training", {}).get("best_last100_episode"),
        )
        run_selector = run["run_id"]

        if evals.get("count", 0) == 0 and best_candidate is not None and not has_active_queue_request(
            queue_rows,
            run["run_id"],
            "best",
        ):
            suggestions.append(f"/eval {run_selector} best 10 queue")

        needs_latest = latest_ep is not None and (
            evals.get("count", 0) == 0 or evals.get("latest_checkpoint_eval") is None
        )
        if needs_latest and not has_active_queue_request(queue_rows, run["run_id"], "latest"):
            suggestions.append(f"/eval {run_selector} latest 10 queue")
    if not suggestions:
        return []
    return ["Suggested manual eval commands"] + suggestions


def render_eval_line(snapshot: dict[str, Any]) -> str:
    evals = snapshot.get("evals", {})
    if evals.get("count", 0) == 0:
        return "Eval: none yet"
    last = evals.get("latest", {})
    latest_ckpt_eval = evals.get("latest_checkpoint_eval")
    newest_ckpt_eval = evals.get("newest_checkpoint_eval")
    best = evals.get("best", {})

    last_ckpt = last.get("checkpoint_id") or last.get("checkpoint") or last.get("checkpoint_episode", "unknown")
    last_mean = fmt_float(safe_float(last.get("mean")), 2)
    best_mean = fmt_float(safe_float(best.get("mean")), 2)
    best_ckpt = best.get("checkpoint_id") or best.get("checkpoint") or best.get("checkpoint_episode", "unknown")

    parts = [f"last {last_ckpt} mean={last_mean} episodes={last.get('episodes', 'N/A')}"]
    if latest_ckpt_eval:
        latest_ckpt = (
            latest_ckpt_eval.get("checkpoint_id")
            or latest_ckpt_eval.get("checkpoint")
            or latest_ckpt_eval.get("checkpoint_episode", "unknown")
        )
        latest_ckpt_mean = fmt_float(safe_float(latest_ckpt_eval.get("mean")), 2)
        parts.append(f"latest ckpt {latest_ckpt} mean={latest_ckpt_mean}")
    elif newest_ckpt_eval:
        newest_ckpt = (
            newest_ckpt_eval.get("checkpoint_id")
            or newest_ckpt_eval.get("checkpoint")
            or newest_ckpt_eval.get("checkpoint_episode", "unknown")
        )
        newest_ckpt_mean = fmt_float(safe_float(newest_ckpt_eval.get("mean")), 2)
        parts.append(f"latest ckpt not evaluated; newest ckpt eval {newest_ckpt} mean={newest_ckpt_mean}")
    else:
        parts.append("latest ckpt not evaluated")
    parts.append(f"best {best_ckpt} mean={best_mean}")
    return "Eval: " + " | ".join(parts)


def render_best(snapshots: list[dict[str, Any]]) -> str:
    lines = ["Best checkpoints:"]
    for snapshot in snapshots:
        run = snapshot["run"]
        lines.append(f"{run['run_id']} ({run['game']}):")
        if run.get("type") == "sb3":
            ckpt = snapshot.get("checkpoints", {})
            lines.append(f"  latest checkpoint: {fmt_steps(ckpt.get('latest_steps'))} steps")
        else:
            train = snapshot.get("training", {})
            ckpt = snapshot.get("checkpoints", {})
            lines.append(
                f"  best train last100: {fmt_float(train.get('best_last100'), 2)} "
                f"at episode {train.get('best_last100_episode', 'N/A')}"
            )
            lines.append(f"  latest checkpoint: ep{ckpt.get('latest_episode', 'N/A')}")
        lines.append(f"  {render_eval_line(snapshot)}")
    return "\n".join(lines)


def render_archive(snapshots: list[dict[str, Any]]) -> str:
    archived = [snapshot for snapshot in snapshots if is_archived_run(snapshot["run"])]
    if not archived:
        return "Archive: no archived runs"
    lines = ["Archived diagnostic runs:"]
    for snapshot in archived:
        run = snapshot["run"]
        lines.append("")
        lines.append(run_display_name(run))
        current = snapshot.get("current_step")
        target = safe_int(run.get("target_steps"))
        unit = run.get("step_unit", "steps")
        lines.append(f"Stopped at: {fmt_steps(current)} / {fmt_steps(target)} {unit}")
        lines.append(render_eval_line(snapshot))
        lines.append(f"Verdict: {render_verdict_line(snapshot)}")
    return "\n".join(lines).rstrip()


def render_report(snapshots: list[dict[str, Any]], resources: dict[str, dict[str, float | None]], registry_source: str) -> str:
    lines = [
        "# DQN Research Report",
        "",
        f"Generated: {local_timestamp()}",
        f"Registry: `{registry_source}`",
        "",
        "## Cluster",
        "",
    ]
    for line in render_resources(resources, [s["run"] for s in snapshots], live_jobs=read_live_slurm_jobs()).splitlines()[1:]:
        lines.append(f"- {line}")
    lines.extend(["", "## Runs", ""])
    for snapshot in snapshots:
        run = snapshot["run"]
        lines.append(f"### {run['run_id']}")
        lines.append("")
        for line in render_run_block(snapshot)[1:]:
            lines.append(f"- {line}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_help(topic: str | None = None) -> str:
    key = (topic or "overview").lower()
    if key in {"overview", "help"}:
        return """DQN Research Bot Help

This bot monitors the Atari DQN replication project.

There are two layers:

1. Machine status
Use poormans in a terminal for the full cluster dashboard.
This bot focuses on research status, not the visual machine cards.

2. Research status
The bot reads the canonical registry:
ofi1:/home/uace/dqn/atari/run_registry.csv

Main commands:
/status       compact research state
/resources    CPU/RAM/load by node
/runs         registered experiments
/archive      stopped diagnostic runs
/best RUN     latest, best_train, best_eval
/eval RUN latest 10
/eval RUN best 10
/eval RUN best_train 10
/eval RUN best_eval 30
/eval RUN ep14000 10
/eval RUN latest 10 queue
/queue        pending evals
/queue drain  run one queued eval if lowkey-safe
/tail RUN 50  last log lines
/sb3          SB3-only status
/report       markdown summary
/history      old reward history from bot
/plot RUN     old reward plot command

Help pages:
/help status
/help eval
/help runs
/help archive
/help queue
/help auto-eval

Research rule:
Training reward is a health signal.
Eval score is the evidence ledger."""

    if key == "eval":
        return """Eval Help

Eval commands:
/eval breakout latest 10
/eval breakout best 10
/eval breakout best_train 10
/eval breakout best_eval 30
/eval breakout ep14000 10
/eval breakout final 30

Checkpoint selectors:
latest: newest checkpoint saved by training.
best: short alias for best_train.
best_train: checkpoint nearest the best training last100 reward.
best_eval: checkpoint with the best previous eval mean.
final: final model at end of training.
epNNN: specific checkpoint episode.

Episode counts:
10 episodes: live signal.
30 episodes: report-grade check.
100 episodes: stronger final evidence.

Safety:
Default eval runs low-priority with nice/ionice and 2 Torch threads.
It refuses only if owner CPU > 98% or RAM > 98%.
Use strict for 85% CPU / 90% RAM behavior.
Use force to run anyway.
Use queue to enqueue if unsafe:
/eval breakout latest 10 queue

Placement:
Scratch DQN eval currently runs on the owner node because the checkpoint lives there.
Cross-node copy-eval is the next phase."""

    if key in {"status", "verdict", "verdicts"}:
        return """Status Help

/status is a research briefing, not a replacement for poormans.
Use it to answer: what should I believe about learning right now?

/status shows:
active runs
progress by frames/timesteps
latest checkpoint
training last100 and best training last100
drop from best training window
best checkpoint candidate
last eval and best eval
conservative verdict
log freshness
eval capacity and queue count

How to read it:
Drop from best is the smoking-gun training signal.
Example: -70% means the current last100 is far below the best observed training window.

Best checkpoint candidate is the saved checkpoint nearest best_train.
That is usually the first checkpoint to evaluate when training looks degraded.

Eval: none yet means no research evidence exists yet.
The next move is usually:
/eval RUN best 10 queue
/eval RUN latest 10 queue

Verdicts:
healthy: running and no immediate issue detected.
learning: eval is positive and training is near its best window.
weak: eval exists but score is low.
stable but weak: eval is consistently positive but best score remains below 20.
unstable: latest eval regressed hard versus best eval.
collapsed_candidate: repeated eval evidence is bad; not training-only.
needs_eval: training exists but eval evidence is missing.
finished: target reached or final checkpoint exists.
error: stale logs, parse errors, or bad eval ledger rows.

Important:
The bot should not call something collapsed from one bad training window.
Actual collapse needs eval evidence."""

    if key == "runs":
        return """Runs Help

/runs lists experiments from the canonical registry.

Registry fields include:
run_id
algo
env_id
owner_node
ssh_target
venv_path
eval_script
checkpoint_pattern
metric_type

Why this matters:
Scratch DQN uses q_net_epNNN.pt and rewards.csv.
SB3 uses .zip checkpoints and monitor.csv.
The registry keeps those protocols explicit instead of forcing one brittle pattern."""

    if key == "archive":
        return """Archive Help

/archive shows stopped or archived diagnostic runs.

Default /status hides these runs so hourly alerts only show active work, active evals, and urgent failures.

Use /status all when you want the active briefing plus archived context.
Use /archive when you only want the stopped diagnostic runs."""

    if key == "queue":
        return """Queue Help

/queue shows pending/running/done eval requests.
/queue drain runs one pending eval if lowkey-safe.
/queue drain --force runs one pending eval even if hot.

Queue file:
ofi1:/home/uace/dqn/atari/eval_queue.csv

Queue rows include:
created_at
request_id
run_id
checkpoint_selector
episodes
epsilon
requested_by
trigger
milestone_step
status
attempts
last_error

Use this when eval should happen eventually but the node is too hot right now."""

    if key in {"auto", "auto-eval", "autoeval"}:
        return """Auto-Eval Help

Automatic eval is milestone-based, not constant.

Scratch DQN 50M milestones:
5M, 10M, 15M, 20M, 25M, 30M, 35M, 40M, 45M, 50M frames.

At each new milestone:
queue eval latest checkpoint for 10 episodes.
drain one queued eval if lowkey-safe.
append results to eval_results.csv.

The first auto-eval run marks already-passed milestones as seen.
That prevents stale backfill spam.

Manual command:
/auto-eval
/auto-eval --no-drain

Final evidence after training should still include:
final checkpoint, 30 episodes.
best_train checkpoint, 30 episodes.
best_eval checkpoint, optionally 100 episodes."""

    return f"Unknown help topic: {topic}\nTry /help, /help eval, /help status, /help runs, /help queue, or /help auto-eval"


def resolve_run(runs: list[dict[str, str]], name: str) -> dict[str, str]:
    needle = name.lower().replace("-", "_")
    exact = [r for r in runs if r["run_id"].lower() == needle]
    if exact:
        return exact[0]
    contains = [
        r
        for r in runs
        if needle in r["run_id"].lower().replace("-", "_")
        or needle in r["game"].lower().replace("-", "_")
    ]
    if len(contains) == 1:
        return contains[0]
    if contains:
        running = [r for r in contains if r.get("status") == "running"]
        return running[0] if running else contains[0]
    raise SystemExit(f"unknown run: {name}")


def select_snapshots(
    runner: ShellRunner,
    runs: list[dict[str, str]],
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = [resolve_run(runs, name) for name in names] if names else runs
    return [build_snapshot(runner, run) for run in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=None)
    sub = parser.add_subparsers(dest="command")
    p_status = sub.add_parser("status")
    p_status.add_argument("scope", nargs="?", choices=["active", "all"], default="active")
    sub.add_parser("resources")
    sub.add_parser("runs")
    p_tail = sub.add_parser("tail")
    p_tail.add_argument("run")
    p_tail.add_argument("lines", nargs="?", type=int, default=80)
    p_best = sub.add_parser("best")
    p_best.add_argument("run", nargs="?")
    sub.add_parser("archive")
    p_sb3 = sub.add_parser("sb3")
    p_sb3.add_argument("run", nargs="?")
    p_help = sub.add_parser("help")
    p_help.add_argument("topic", nargs="?")
    sub.add_parser("report")
    args = parser.parse_args()

    command = args.command or "status"
    registry_source = str(args.registry) if args.registry else canonical_registry_label()
    runs = load_registry(args.registry)
    runner = ShellRunner()

    if command == "help":
        print(render_help(args.topic))
        return

    if command == "runs":
        print(render_runs(runs))
        return

    if command == "resources":
        prom = choose_prom()
        print(render_resources(load_cluster_resources(prom), runs, live_jobs=read_live_slurm_jobs()))
        return

    if command == "tail":
        run = resolve_run(runs, args.run)
        print(runner.tail_file(run, run["log"], args.lines), end="")
        return

    if command == "best":
        names = [args.run] if args.run else None
        print(render_best(select_snapshots(runner, runs, names)))
        return

    if command == "archive":
        print(render_archive(select_snapshots(runner, runs)))
        return

    if command == "sb3":
        sb3_runs = [r for r in runs if r.get("type") == "sb3"]
        if args.run:
            sb3_runs = [resolve_run(sb3_runs, args.run)]
        snapshots = [build_snapshot(runner, r) for r in sb3_runs]
        print("\n\n".join("\n".join(render_run_block(s)) for s in snapshots))
        return

    prom = choose_prom()
    resources = load_cluster_resources(prom)
    snapshots = select_snapshots(runner, runs)
    if command == "report":
        print(render_report(snapshots, resources, registry_source))
    else:
        print(render_status(snapshots, resources, registry_source, show_all=getattr(args, "scope", "active") == "all"))


if __name__ == "__main__":
    main()
