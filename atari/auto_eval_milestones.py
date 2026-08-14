#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from eval_checkpoint import (
    CANONICAL_NODE,
    enqueue_eval,
    read_remote_file,
    run_eval_request,
    run_remote_python,
    utc_stamp,
)
from research_status import ShellRunner, build_snapshot, load_registry, safe_float, safe_int


STATE_PATH = os.environ.get("DQN_RESEARCH_AUTO_EVAL_STATE", "/home/uace/dqn/atari/auto_eval_state.csv")
LOCK_DIR = os.environ.get("DQN_RESEARCH_AUTO_EVAL_LOCK", "/tmp/dqn_auto_eval_milestones.lock")
STATE_FIELDS = ["run_id", "last_seen_milestone", "updated_at"]
SUMMARY_DIR = os.environ.get("DQN_RESEARCH_AUTO_EVAL_SUMMARIES", "/home/uace/dqn/atari/auto_eval_summaries")
DEFAULT_NOTIFY_COMMAND = f"{sys.executable} {Path(__file__).with_name('telegram_notify_report.py')}"
NOTIFY_COMMAND = os.environ.get("DQN_RESEARCH_NOTIFY_CMD", DEFAULT_NOTIFY_COMMAND)
FINAL_EVAL_EPISODES = 30


def read_state(runner: ShellRunner) -> dict[str, dict[str, str]]:
    text = read_remote_file(runner, CANONICAL_NODE, STATE_PATH, 15)
    if not text.strip():
        return {}
    rows = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("run_id"):
            rows[row["run_id"]] = {field: row.get(field, "") for field in STATE_FIELDS}
    return rows


def write_state(runner: ShellRunner, rows_by_run: dict[str, dict[str, str]]) -> None:
    rows = [{field: row.get(field, "") for field in STATE_FIELDS} for row in rows_by_run.values()]
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
    result = run_remote_python(runner, CANONICAL_NODE, script, [STATE_PATH, json.dumps(STATE_FIELDS), json.dumps(rows)], 20)
    if result.returncode != 0:
        raise SystemExit(f"failed to write auto-eval state: {result.stderr.strip()}")


def acquire_lock(runner: ShellRunner) -> bool:
    result = runner.run_shell(CANONICAL_NODE, f"mkdir {LOCK_DIR!r} 2>/dev/null && printf locked || true", 10)
    return result.stdout.strip() == "locked"


def release_lock(runner: ShellRunner) -> None:
    runner.run_shell(CANONICAL_NODE, f"rmdir {LOCK_DIR!r} 2>/dev/null || true", 10)


def current_milestone(current_step: int | None, every: int, target: int) -> int:
    if current_step is None or every <= 0:
        return 0
    milestone = (current_step // every) * every
    if target > 0:
        milestone = min(milestone, target)
    return int(milestone)


def remote_file_exists(runner: ShellRunner, run: dict[str, str], path: str) -> bool:
    result = runner.run_shell(run, f"test -f {shlex.quote(path)} && printf yes || true", 10)
    return result.stdout.strip() == "yes"


def write_remote_text(runner: ShellRunner, run: dict[str, str], path: str, text: str) -> None:
    script = (
        "import os,sys;"
        "path=sys.argv[1]; text=sys.argv[2];"
        "os.makedirs(os.path.dirname(path), exist_ok=True);"
        "tmp=path+'.tmp';"
        "open(tmp,'w').write(text);"
        "os.replace(tmp,path)"
    )
    result = run_remote_python(runner, run, script, [path, text], 20)
    if result.returncode != 0:
        raise RuntimeError(f"failed to write {path}: {result.stderr.strip()}")


def claim_finalization(runner: ShellRunner, run: dict[str, str]) -> str:
    """Atomically claim a completed run and expose the requested marker files."""
    run_dir = run["run_dir"]
    done = f"{run_dir}/AUTOEVAL_DONE"
    running = f"{run_dir}/AUTOEVAL_RUNNING"
    lock = f"{run_dir}/.autoeval_final.lock"
    command = (
        f"if test -e {shlex.quote(done)}; then printf done; "
        f"elif mkdir {shlex.quote(lock)} 2>/dev/null; then "
        f"date -u +%Y-%m-%dT%H:%M:%SZ > {shlex.quote(running)}; printf claimed; "
        f"else printf running; fi"
    )
    return runner.run_shell(run, command, 15).stdout.strip()


def release_finalization(runner: ShellRunner, run: dict[str, str], done: bool) -> None:
    run_dir = run["run_dir"]
    running = f"{run_dir}/AUTOEVAL_RUNNING"
    lock = f"{run_dir}/.autoeval_final.lock"
    done_path = f"{run_dir}/AUTOEVAL_DONE"
    command = f"rm -f {shlex.quote(running)}; "
    if done:
        command += f"date -u +%Y-%m-%dT%H:%M:%SZ > {shlex.quote(done_path)}; "
    command += f"rmdir {shlex.quote(lock)} 2>/dev/null || true"
    runner.run_shell(run, command, 15)


def successful_auto_eval(snapshot: dict, trigger: str) -> dict | None:
    matches = [
        row
        for row in snapshot.get("evals", {}).get("rows", [])
        if row.get("trigger") == trigger
        and safe_int(row.get("episodes"), 0) == FINAL_EVAL_EPISODES
    ]
    return matches[-1] if matches else None


def run_final_eval(
    runs: list[dict[str, str]],
    run: dict[str, str],
    selector: str,
    trigger: str,
) -> tuple[bool, str]:
    row = {
        "run_id": run["run_id"],
        "checkpoint_selector": selector,
        "episodes": str(FINAL_EVAL_EPISODES),
        "epsilon": "0.05",
        "requested_by": "auto_eval",
        "trigger": trigger,
        "milestone_step": "",
    }
    ok, output, _ = run_eval_request(runs, row)
    return ok, output


def find_eval_row(snapshot: dict, trigger: str) -> dict | None:
    return successful_auto_eval(snapshot, trigger)


def render_final_summary(
    snapshot: dict,
    final_row: dict | None,
    best_row: dict | None,
    best_selector: str,
) -> str:
    run = snapshot["run"]
    training = snapshot.get("training", {})
    evals = snapshot.get("evals", {})
    current_step = snapshot.get("current_step")
    step_unit = run.get("step_unit")
    final_mean = safe_float(final_row.get("mean")) if final_row else None
    best_mean = safe_float(best_row.get("mean")) if best_row else None

    if final_mean is not None and best_mean is not None and final_mean < 0.5 * best_mean:
        verdict = "learned, then degraded / unstable"
    elif final_mean is not None and best_mean is not None:
        verdict = "final policy retained competitive performance"
    else:
        verdict = "final evaluation incomplete"

    def score(row: dict | None) -> str:
        if row is None:
            return "missing"
        return f"mean {row.get('mean', '?')} over {row.get('episodes', '?')} eps ({row.get('checkpoint_id', '?')})"

    progress_label = "agent decisions" if step_unit == "decisions" else "raw frames"
    progress_lines = [
        f"- {progress_label}: {current_step if current_step is not None else 'unknown'}",
    ]
    if step_unit == "decisions":
        progress_lines.append(
            f"- raw emulator frames: {snapshot.get('env_step', 'unknown')}"
        )

    lines = [
        f"DQN run finished: {run['run_id']}",
        "",
        "Training:",
        *progress_lines,
        f"- episodes: {training.get('latest_episode', 'unknown')}",
        f"- best last100: {training.get('best_last100', 'unknown')} @ ep{training.get('best_last100_episode', 'unknown')}",
        f"- final last100: {training.get('last100', 'unknown')}",
        "",
        "Evaluation:",
        f"- final checkpoint: {score(final_row)}",
        f"- {best_selector} checkpoint: {score(best_row)}",
        f"- successful eval ledger rows: {evals.get('count', 0)}",
        "",
        f"Verdict: {verdict}",
        "",
        "Artifacts:",
        f"- run dir: {run['run_dir']}",
        f"- metadata: {run['run_dir']}/RUN_METADATA.json",
        f"- final checkpoint: {run['run_dir']}/checkpoint_final.pt",
        f"- eval results: {run['run_dir']}/eval_results.csv",
    ]
    return "\n".join(lines) + "\n"


def pct_drop(best: float | None, final: float | None) -> str:
    if best is None or best <= 0 or final is None:
        return "n/a"
    return f"-{(1.0 - final / best) * 100:.1f}%"


def compact_run_name(run_id: str, game: str) -> str:
    prefix = game.lower() + "_"
    if run_id.lower().startswith(prefix):
        return run_id[len(prefix) :].replace("_", " ")
    return run_id.replace("_", " ")


def render_compact_autoeval_report(
    run_id: str,
    game: str,
    progress: int | None,
    step_unit: str | None,
    episodes: int | None,
    best_eval_mean: float | None,
    best_eval_label: str,
    final_eval_mean: float | None,
    best_last100: float | None,
    best_last100_ep: int | None,
    final_last100: float | None,
) -> str:
    drop = pct_drop(best_eval_mean, final_eval_mean)
    if best_eval_mean is not None and final_eval_mean is not None and best_eval_mean > 0:
        ratio = final_eval_mean / best_eval_mean
        verdict = "learned, then collapsed" if ratio < 0.1 else "learned, then degraded"
    else:
        verdict = "final evaluation incomplete"

    progress_text = f"{progress / 1_000_000:.1f}M" if progress is not None else "unknown"
    progress_label = "agent decisions" if step_unit == "decisions" else "frames"
    episodes_text = f"{episodes:,}" if episodes is not None else "unknown"
    best_eval_text = f"{best_eval_mean:.2f}" if best_eval_mean is not None else "n/a"
    final_eval_text = f"{final_eval_mean:.2f}" if final_eval_mean is not None else "n/a"
    best_train_text = f"{best_last100:.2f}" if best_last100 is not None else "n/a"
    final_train_text = f"{final_last100:.2f}" if final_last100 is not None else "n/a"
    best_train_ep_text = str(best_last100_ep) if best_last100_ep is not None else "n/a"

    return "\n".join(
        [
            f"DQN finished · {game} {compact_run_name(run_id, game)}",
            "",
            f"{progress_text} {progress_label} · {episodes_text} episodes",
            f"Best eval: {best_eval_text} @ {best_eval_label}",
            f"Final eval: {final_eval_text}",
            f"Eval collapse: {drop}",
            "",
            "Training:",
            f"best last100 {best_train_text} @ ep{best_train_ep_text}",
            f"final last100 {final_train_text}",
            "",
            f"Verdict: {verdict}",
            "Next: audit reward clipping + train frequency before more training",
        ]
    ) + "\n"


def notify_summary(summary_path: str) -> str:
    try:
        proc = subprocess.run(
            shlex.split(NOTIFY_COMMAND) + [summary_path],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except OSError as exc:
        return f"notification hook failed to start: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        return f"notification hook failed: {detail[:240]}"
    return "notification hook completed"


def finalize_completed_run(runner: ShellRunner, runs: list[dict[str, str]], run: dict[str, str]) -> str | None:
    final_path = f"{run['run_dir']}/q_net_final.pt"
    if not remote_file_exists(runner, run, final_path):
        return None

    claim = claim_finalization(runner, run)
    if claim == "done":
        return None
    if claim != "claimed":
        return f"{run['run_id']}: final auto-eval already running"

    completed = False
    try:
        before = build_snapshot(runner, run)
        best_selector = "best_eval" if before.get("evals", {}).get("count", 0) else "best_train"

        if not successful_auto_eval(before, "final_auto_final"):
            ok, output = run_final_eval(runs, run, "final", "final_auto_final")
            if not ok:
                return f"{run['run_id']}: final 30-episode eval failed: {output[-280:]}"

        after_final = build_snapshot(runner, run)
        if not successful_auto_eval(after_final, "final_auto_best"):
            ok, output = run_final_eval(runs, run, best_selector, "final_auto_best")
            if not ok:
                return f"{run['run_id']}: {best_selector} 30-episode eval failed: {output[-280:]}"

        completed_snapshot = build_snapshot(runner, run)
        final_row = find_eval_row(completed_snapshot, "final_auto_final")
        best_row = find_eval_row(completed_snapshot, "final_auto_best")
        summary = render_final_summary(
            completed_snapshot,
            final_row,
            best_row,
            best_selector,
        )
        write_remote_text(runner, run, f"{run['run_dir']}/AUTOEVAL_SUMMARY.txt", summary)
        canonical_summary = f"{SUMMARY_DIR}/{run['run_id']}.txt"
        write_remote_text(runner, CANONICAL_NODE, canonical_summary, summary)
        training = completed_snapshot.get("training", {})
        compact_summary = render_compact_autoeval_report(
            run["run_id"],
            run["game"],
            completed_snapshot.get("current_step"),
            run.get("step_unit"),
            safe_int(training.get("latest_episode"), None),
            safe_float(best_row.get("mean")) if best_row else None,
            best_row.get("checkpoint_id", "best") if best_row else "best",
            safe_float(final_row.get("mean")) if final_row else None,
            safe_float(training.get("best_last100")),
            safe_int(training.get("best_last100_episode"), None),
            safe_float(training.get("last100")),
        )
        canonical_compact_summary = f"{SUMMARY_DIR}/{run['run_id']}.telegram.txt"
        write_remote_text(runner, CANONICAL_NODE, canonical_compact_summary, compact_summary)
        completed = True
        notification = notify_summary(canonical_compact_summary)
        return f"{run['run_id']}: final auto-eval complete; {notification}\n{compact_summary.rstrip()}"
    finally:
        release_finalization(runner, run, done=completed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-drain", action="store_true")
    parser.add_argument("--drain-limit", type=int, default=1)
    parser.add_argument("--force-drain", action="store_true")
    args = parser.parse_args()

    runner = ShellRunner()
    if not acquire_lock(runner):
        print("auto-eval already running")
        return

    queued = []
    runs: list[dict[str, str]] = []
    try:
        state = read_state(runner)
        runs = load_registry(None)
        for run in runs:
            if not run.get("eval_script"):
                continue
            if run.get("status") not in {"running", "finished"}:
                continue
            every = safe_int(run.get("eval_every_steps"), 0)
            target = safe_int(run.get("target_steps"), 0)
            if every <= 0:
                continue

            snapshot = build_snapshot(runner, run)
            milestone = current_milestone(snapshot.get("current_step"), every, target)
            if milestone <= 0:
                continue

            row = state.get(run["run_id"])
            if row is None:
                state[run["run_id"]] = {
                    "run_id": run["run_id"],
                    "last_seen_milestone": str(milestone),
                    "updated_at": utc_stamp(),
                }
                continue

            last_seen = safe_int(row.get("last_seen_milestone"), 0)
            if milestone <= last_seen:
                continue

            qrow = enqueue_eval(
                runner,
                run["run_id"],
                "latest",
                10,
                0.05,
                "auto_eval",
                "milestone",
                str(milestone),
            )
            queued.append(qrow)
            row["last_seen_milestone"] = str(milestone)
            row["updated_at"] = utc_stamp()

        write_state(runner, state)
    finally:
        release_lock(runner)

    if queued:
        for row in queued:
            print(
                f"queued {row['request_id']} {row['run_id']} "
                f"{row['checkpoint_selector']} {row['episodes']}ep milestone={row['milestone_step']}"
            )
    else:
        print("no new eval milestones")

    final_reports = []
    for run in runs:
        if run.get("type") != "scratch_dqn":
            continue
        if run.get("status") not in {"running", "finished"}:
            continue
        try:
            report = finalize_completed_run(runner, runs, run)
        except Exception as exc:
            report = f"{run['run_id']}: final auto-eval setup failed: {exc}"
        if report:
            final_reports.append(report)

    if final_reports:
        print("\n\n".join(final_reports))

    if not args.no_drain:
        cmd = [str(Path(__file__).resolve().with_name("eval_checkpoint.py")), "drain", "--limit", str(args.drain_limit)]
        if args.force_drain:
            cmd.append("--force")
        proc = subprocess.run([sys.executable] + cmd, text=True, capture_output=True, check=False)
        out = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
        if out:
            print(out)


if __name__ == "__main__":
    main()
