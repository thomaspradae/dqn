import argparse
import csv
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from deepmind_rmsprop import DeepMindRMSprop
from network import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gym.register_envs(ale_py)

RESIZE_INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
}


def set_global_seed(seed):
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path):
    path = Path(path)
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "no_git"


def get_git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return None


def package_version(module):
    return getattr(module, "__version__", "unknown")


def write_run_metadata(outdir, args):
    code_files = [
        "train_nature.py",
        "network.py",
        "eval.py",
        "eval_checkpoint.py",
    ]
    metadata = {
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "argv": sys.argv,
        "args": vars(args),
        "seed": args.seed,
        "env_id": args.env_id,
        "budget_unit": (
            "agent_decisions"
            if args.max_agent_steps_total is not None
            else "raw_emulator_frames"
            if args.max_raw_env_frames_total is not None
            else "episodes"
        ),
        "budget_agent_decisions": args.max_agent_steps_total,
        "budget_raw_emulator_frames": (
            args.max_agent_steps_total * args.frame_skip
            if args.max_agent_steps_total is not None
            else args.max_raw_env_frames_total
        ),
        "optimizer": {
            "name": args.optimizer,
            "alpha": 0.95,
            "eps": 0.01,
            "momentum": 0.0 if args.optimizer == "deepmind_rmsprop" else 0.95,
            "centered": args.optimizer == "deepmind_rmsprop",
            "epsilon_inside_sqrt": args.optimizer == "deepmind_rmsprop",
        },
        "resize_interpolation": args.resize_interpolation,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "code_sha256": {path: sha256_file(path) for path in code_files},
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "gymnasium_version": package_version(gym),
        "ale_py_version": package_version(ale_py),
        "cv2_version": package_version(cv2),
        "numpy_version": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "replay_buffer_checkpointed": False,
        "resume_supported": False,
        "notes": [
            "Replay buffer is not checkpointed; checkpoint dict is diagnostic/resumable metadata, not exact replay-state resume.",
            "Evaluation is intentionally watcher/queue based, not inline inside train_nature.py.",
        ],
    }
    metadata_path = outdir / "RUN_METADATA.json"
    legacy_path = outdir / "run_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with legacy_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def run_dir_has_training_artifacts(outdir):
    artifact_patterns = (
        "q_net_ep*.pt",
        "checkpoint_ep*.pt",
        "q_net_final.pt",
        "checkpoint_final.pt",
        "rewards.csv",
        "checkpoints.csv",
        "eval_results.csv",
    )
    return [path for pattern in artifact_patterns for path in outdir.glob(pattern)]


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def preprocess_frame(obs, previous_obs=None, resize_interpolation="area"):
    if previous_obs is not None:
        obs = np.maximum(obs, previous_obs)

    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
    resized = cv2.resize(
        gray,
        (84, 84),
        interpolation=RESIZE_INTERPOLATIONS[resize_interpolation],
    )
    return resized


class ReplayMemory:
    def __init__(self, capacity, frame_shape=(84, 84), history_len=4):
        if capacity <= history_len:
            raise ValueError("replay capacity must be larger than the frame history length")

        self.capacity = capacity
        self.history_len = history_len
        self.frames = np.empty((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.bool_)
        self.frame_valid = np.zeros(capacity, dtype=np.bool_)
        self.frame_ids = np.full(capacity, -1, dtype=np.int64)
        self.episode_starts = np.zeros(capacity, dtype=np.bool_)
        self.transition_valid = np.zeros(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.num_transitions = 0
        self.next_frame_id = 0
        self.last_frame_index = None

    def __len__(self):
        return self.num_transitions

    def start_episode(self, frame):
        for i in range(self.history_len):
            self.last_frame_index = self._store_frame(frame, episode_start=i == 0)

        return self.state_at(self.last_frame_index)

    def append(self, action, reward, next_frame, terminal):
        if self.last_frame_index is None:
            raise RuntimeError("start_episode must be called before append")

        source_index = self.last_frame_index
        next_index = self._store_frame(next_frame, episode_start=False)
        self.actions[source_index] = action
        self.rewards[source_index] = reward
        self.terminals[source_index] = terminal
        self._set_transition_valid(source_index, True)
        self.last_frame_index = next_index
        return self.state_at(next_index)

    def sample(self, batch_size):
        indices = []
        attempts = 0
        max_attempts = batch_size * 100
        sample_high = self.capacity if self.size == self.capacity else self.size

        while len(indices) < batch_size and attempts < max_attempts:
            index = random.randrange(sample_high)
            if self._is_valid_sample(index):
                indices.append(index)
            attempts += 1

        if len(indices) < batch_size:
            candidates = [
                int(index)
                for index in np.flatnonzero(self.transition_valid)
                if self._is_valid_sample(int(index))
            ]
            if len(candidates) < batch_size:
                raise RuntimeError("not enough valid replay transitions to sample a batch")
            indices.extend(random.sample(candidates, batch_size - len(indices)))

        indices = np.array(indices, dtype=np.int64)
        next_indices = (indices + 1) % self.capacity

        states = np.stack([self.state_at(index) for index in indices])
        next_states = np.stack([self.state_at(index) for index in next_indices])

        return (
            states,
            self.actions[indices],
            self.rewards[indices],
            next_states,
            self.terminals[indices],
        )

    def state_at(self, index):
        frame_indices = self._state_indices(index)
        return self.frames[frame_indices]

    def _store_frame(self, frame, episode_start):
        index = self.position

        self._set_transition_valid(index, False)
        self._set_transition_valid((index - 1) % self.capacity, False)

        self.frames[index] = frame
        self.frame_valid[index] = True
        self.frame_ids[index] = self.next_frame_id
        self.episode_starts[index] = episode_start
        self.next_frame_id += 1

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def _set_transition_valid(self, index, is_valid):
        if self.transition_valid[index] == is_valid:
            return

        self.transition_valid[index] = is_valid
        self.num_transitions += 1 if is_valid else -1

    def _is_valid_sample(self, index):
        if not self.transition_valid[index]:
            return False

        next_index = (index + 1) % self.capacity
        state_indices = self._state_indices(index)
        next_state_indices = self._state_indices(next_index)

        if not self.frame_valid[state_indices].all():
            return False
        if not self.frame_valid[next_state_indices].all():
            return False

        if not self._has_consecutive_history(index):
            return False
        if self.frame_ids[next_index] != self.frame_ids[index] + 1:
            return False

        if self.episode_starts[state_indices[1:]].any():
            return False

        if not self.terminals[index]:
            if not self._has_consecutive_history(next_index):
                return False
            if self.episode_starts[next_state_indices[1:]].any():
                return False

        return True

    def _state_indices(self, index):
        offsets = np.arange(self.history_len - 1, -1, -1, dtype=np.int64)
        return (index - offsets) % self.capacity

    def _has_consecutive_history(self, index):
        frame_indices = self._state_indices(index)
        expected_ids = np.arange(
            self.frame_ids[index] - self.history_len + 1,
            self.frame_ids[index] + 1,
            dtype=np.int64,
        )
        return np.array_equal(self.frame_ids[frame_indices], expected_ids)


def clipped_td_error_loss(predictions, targets):
    td_errors = predictions - targets
    abs_errors = td_errors.abs()
    quadratic = torch.minimum(abs_errors, torch.ones_like(abs_errors))
    linear = abs_errors - quadratic
    return (0.5 * quadratic.pow(2) + linear).mean()


def clip_transition_reward(total_raw_reward):
    """Clip the action-repeat return before it enters replay memory."""
    return max(-1.0, min(1.0, float(total_raw_reward)))


def epsilon_for_agent_step(args, agent_step):
    """Match DeepMind's epsilon schedule: decay after learning starts."""
    decay_progress = max(0, agent_step - args.learning_starts)
    progress = min(decay_progress / args.epsilon_decay, 1.0)
    return args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)


def should_learn(perceived_step, learning_starts, train_freq):
    """Match DeepMind's pre-increment `numSteps` update check."""
    return perceived_step > learning_starts and perceived_step % train_freq == 0


def should_update_target(agent_step, target_update_freq):
    """Mirror DeepMind's `numSteps % target_q == 1` target-copy schedule."""
    return agent_step % target_update_freq == 1


def counter_smoke_summary(agent_steps, frame_skip, learning_starts, train_freq, target_update_freq, trace=False):
    update_pre_increment_steps = [
        step for step in range(agent_steps) if should_learn(step, learning_starts, train_freq)
    ]
    target_post_increment_steps = [
        step for step in range(1, agent_steps + 1) if should_update_target(step, target_update_freq)
    ]
    summary = {
        "agent_decisions": agent_steps,
        "nominal_raw_action_frames": agent_steps * frame_skip,
        "optimizer_updates": len(update_pre_increment_steps),
        "target_copies": len(target_post_increment_steps),
    }
    if trace:
        summary["update_pre_increment_steps"] = update_pre_increment_steps
        summary["target_post_increment_steps"] = target_post_increment_steps
    return summary


def build_optimizer(args, parameters):
    if args.optimizer == "deepmind_rmsprop":
        return DeepMindRMSprop(parameters, lr=args.lr, alpha=0.95, eps=0.01)
    return optim.RMSprop(
        parameters,
        lr=args.lr,
        alpha=0.95,
        eps=0.01,
        momentum=0.95,
    )


def get_noop_action(env):
    get_action_meanings = getattr(env.unwrapped, "get_action_meanings", None)
    if get_action_meanings is None:
        return 0

    action_meanings = get_action_meanings()
    if "NOOP" in action_meanings:
        return action_meanings.index("NOOP")

    return 0


def reset_with_noops(env, noop_max, max_noop_steps=None, seed=None):
    if seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=seed)

    if noop_max <= 0 or max_noop_steps == 0:
        return obs, info, 0

    noop_steps = random.randint(1, noop_max)
    if max_noop_steps is not None:
        noop_steps = min(noop_steps, max_noop_steps)

    noop_action = get_noop_action(env)
    steps_taken = 0

    for _ in range(noop_steps):
        obs, _, terminated, truncated, info = env.step(noop_action)
        steps_taken += 1

        if terminated or truncated:
            obs, info = env.reset()

    return obs, info, steps_taken


def get_lives(env):
    ale = getattr(env.unwrapped, "ale", None)
    if ale is None:
        return 0

    return ale.lives()


def save_checkpoint(path, episode, env_step, agent_step, train_step, target_update_count, epsilon, q_net, target_net, optimizer, args):
    torch.save(
        {
            "episode": episode,
            "env_step": env_step,
            "agent_step": agent_step,
            "train_step": train_step,
            "target_update_count": target_update_count,
            "epsilon": epsilon,
            "q_net": q_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_state_dict": q_net.state_dict(),
            "target_model_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "rng": rng_state(),
            "schema_version": 2,
        },
        path,
    )


def append_checkpoint_manifest(outdir, episode, env_step, agent_step, train_step, epsilon, q_net_path, checkpoint_path):
    manifest_path = outdir / "checkpoints.csv"
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "episode",
                    "env_step",
                    "agent_step",
                    "train_step",
                    "epsilon",
                    "q_net_path",
                    "checkpoint_path",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                episode,
                env_step,
                agent_step,
                train_step,
                f"{epsilon:.6f}",
                q_net_path,
                checkpoint_path,
            ]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--outdir", type=str, default="runs/atari")
    parser.add_argument("--run-dir", dest="outdir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument(
        "--optimizer",
        choices=("pytorch_rmsprop", "deepmind_rmsprop"),
        default="pytorch_rmsprop",
        help="optimizer implementation; the active v3 baseline uses pytorch_rmsprop",
    )
    parser.add_argument(
        "--resize-interpolation",
        choices=tuple(RESIZE_INTERPOLATIONS),
        default="area",
        help="84x84 resize method; released DeepMind code uses bilinear",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-freq", type=int, default=10_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every-episodes",
        dest="checkpoint_every",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument(
        "--max-agent-steps-total",
        type=int,
        default=None,
        help="selected-action / perceived-state budget",
    )
    parser.add_argument(
        "--max-raw-env-frames-total",
        type=int,
        default=None,
        help="raw emulator-frame budget, including reset no-ops",
    )
    parser.add_argument(
        "--counter-smoke-test",
        action="store_true",
        help="print deterministic counter totals and exit without creating an environment",
    )
    parser.add_argument("--trace-counters", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--diagnostics-every-agent-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-existing-run-dir", action="store_true")
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument(
        "--life-loss-terminal",
        dest="life_loss_terminal",
        action="store_true",
    )
    parser.add_argument(
        "--no-life-loss-terminal",
        dest="life_loss_terminal",
        action="store_false",
    )

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--start-env-step", type=int, default=0)
    parser.add_argument("--start-train-step", type=int, default=0)
    parser.set_defaults(life_loss_terminal=True)

    args = parser.parse_args()

    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")
    if args.noop_max < 0:
        parser.error("--noop-max must be >= 0")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.replay_size <= 4:
        parser.error("--replay-size must be greater than 4")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.train_freq < 1:
        parser.error("--train-freq must be >= 1")
    if args.learning_starts < 0:
        parser.error("--learning-starts must be >= 0")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")
    if args.diagnostics_every_agent_steps < 1:
        parser.error("--diagnostics-every-agent-steps must be >= 1")
    if args.max_agent_steps_total is not None and args.max_agent_steps_total < 1:
        parser.error("--max-agent-steps-total must be >= 1")
    if args.max_raw_env_frames_total is not None and args.max_raw_env_frames_total < 1:
        parser.error("--max-raw-env-frames-total must be >= 1")
    if args.max_agent_steps_total is not None and args.max_raw_env_frames_total is not None:
        parser.error("use only one of --max-agent-steps-total or --max-raw-env-frames-total")
    if args.resume is not None:
        parser.error("--resume is disabled because replay memory is not checkpointed")
    if args.start_episode != 0 or args.start_env_step != 0 or args.start_train_step != 0:
        parser.error("--start-* counters are disabled because replay memory is not checkpointed")

    if args.counter_smoke_test:
        if args.max_agent_steps_total is None:
            parser.error("--counter-smoke-test requires --max-agent-steps-total")
        print(
            json.dumps(
                counter_smoke_summary(
                    args.max_agent_steps_total,
                    args.frame_skip,
                    args.learning_starts,
                    args.train_freq,
                    args.target_update_freq,
                    trace=args.trace_counters,
                ),
                sort_keys=True,
            )
        )
        return

    set_global_seed(args.seed)
    torch.set_num_threads(args.threads)

    outdir = Path(args.outdir)
    if outdir.exists() and not args.allow_existing_run_dir:
        artifacts = run_dir_has_training_artifacts(outdir)
        if artifacts:
            names = ", ".join(path.name for path in sorted(artifacts)[:8])
            if len(artifacts) > 8:
                names += ", ..."
            raise SystemExit(
                f"refusing to start in non-empty run dir with training artifacts: {outdir} ({names}). "
                "Use a new --run-dir, or pass --allow-existing-run-dir only for deliberate debugging."
            )
    outdir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(outdir, args)

    env = gym.make(args.env_id, frameskip=1, repeat_action_probability=0.0)
    if args.seed is not None:
        env.action_space.seed(args.seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(args.seed)
    num_actions = env.action_space.n

    q_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = build_optimizer(args, q_net.parameters())
    loss_fn = clipped_td_error_loss

    start_episode = 0
    env_step = 0
    step = 0
    action_step = 0
    target_update_count = 0

    replay_buffer = ReplayMemory(args.replay_size)
    printed_debug_shapes = False
    action_counts = np.zeros(num_actions, dtype=np.int64)
    clipped_reward_counts = {"-1": 0, "0": 0, "+1": 0}
    life_loss_count = 0
    diagnostics = {"loss": [], "td_abs": [], "q_selected": [], "q_max": [], "grad_norm": []}

    rewards_csv = outdir / "rewards.csv"
    diagnostics_csv = outdir / "training_diagnostics.csv"
    csv_mode = "w"
    write_header = True

    last_episode = start_episode - 1
    diagnostics_file = diagnostics_csv.open("w", newline="")
    diagnostics_writer = csv.DictWriter(
        diagnostics_file,
        fieldnames=["agent_step", "env_step", "train_step", "episode", "epsilon", "raw_episode_return", "loss_mean", "td_error_abs_mean", "td_error_abs_p95", "q_selected_mean", "q_max_mean", "q_max_abs_max", "grad_norm_mean", "clip_neg1", "clip_zero", "clip_pos1", "life_loss_count", "target_copy_count", "action_counts"],
    )
    diagnostics_writer.writeheader()

    with open(rewards_csv, csv_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward", "epsilon"])

        for episode in range(start_episode, args.episodes):
            last_episode = episode
            max_noop_steps = None
            if args.max_raw_env_frames_total is not None:
                max_noop_steps = max(args.max_raw_env_frames_total - env_step, 0)

            obs, info, noop_steps = reset_with_noops(
                env,
                args.noop_max,
                max_noop_steps=max_noop_steps,
                seed=args.seed if episode == start_episode else None,
            )
            env_step += noop_steps

            if (
                args.max_raw_env_frames_total is not None
                and env_step >= args.max_raw_env_frames_total
            ):
                break

            state = replay_buffer.start_episode(
                preprocess_frame(obs, resize_interpolation=args.resize_interpolation)
            )
            last_raw_obs = obs
            lives = get_lives(env)
            episode_done = False
            episode_reward = 0.0
            episode_steps = 0
            epsilon = args.epsilon_start

            while not episode_done:
                if (
                    args.max_agent_steps_total is not None
                    and action_step >= args.max_agent_steps_total
                ):
                    episode_done = True
                    break
                if (
                    args.max_raw_env_frames_total is not None
                    and env_step >= args.max_raw_env_frames_total
                ):
                    episode_done = True
                    break

                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    q_values = q_net(state_tensor)

                epsilon = epsilon_for_agent_step(args, action_step)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax(dim=1).item()
                action_counts[action] += 1

                raw_action_reward = 0.0
                previous_raw_obs = last_raw_obs
                env_done = False
                forced_done = False
                life_lost = False

                for _ in range(args.frame_skip):
                    previous_raw_obs = last_raw_obs
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    last_raw_obs = next_obs
                    env_step += 1

                    raw_action_reward += reward

                    if env_step % 1000 == 0:
                        print(
                            f"env_step={env_step} agent_step={action_step} train_step={step} "
                            f"episode={episode} reward={episode_reward + raw_action_reward} "
                            f"epsilon={epsilon:.3f}",
                            flush=True,
                        )

                    env_done = terminated or truncated

                    current_lives = get_lives(env)
                    if args.life_loss_terminal and lives > 0 and current_lives < lives:
                        life_lost = True
                        life_loss_count += 1
                    lives = current_lives

                    if (
                        args.max_raw_env_frames_total is not None
                        and env_step >= args.max_raw_env_frames_total
                    ):
                        forced_done = True

                    if env_done or forced_done or life_lost:
                        break

                episode_steps += 1
                completed_agent_step = action_step + 1
                episode_reward += raw_action_reward
                if (
                    args.max_agent_steps_total is not None
                    and completed_agent_step >= args.max_agent_steps_total
                ):
                    forced_done = True
                episode_done = env_done or forced_done
                transition_terminal = episode_done or life_lost

                if args.max_steps_per_episode is not None and episode_steps >= args.max_steps_per_episode:
                    episode_done = True
                    transition_terminal = True

                next_frame = preprocess_frame(
                    last_raw_obs,
                    previous_raw_obs,
                    resize_interpolation=args.resize_interpolation,
                )
                transition_reward = clip_transition_reward(raw_action_reward)
                clipped_reward_counts[
                    "+1" if transition_reward > 0 else "-1" if transition_reward < 0 else "0"
                ] += 1
                next_state = replay_buffer.append(
                    action,
                    transition_reward,
                    next_frame,
                    transition_terminal,
                )
                if life_lost and not episode_done:
                    next_state = replay_buffer.start_episode(next_frame)

                if (
                    should_learn(action_step, args.learning_starts, args.train_freq)
                    and len(replay_buffer) >= args.batch_size
                ):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(
                        args.batch_size
                    )

                    states = torch.as_tensor(states, dtype=torch.float32, device=device)
                    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
                    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                    next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

                    if args.debug_shapes and not printed_debug_shapes:
                        print(states.shape)
                        print(q_net(states).shape)
                        printed_debug_shapes = True

                    with torch.no_grad():
                        max_next_q = target_net(next_states).max(dim=1).values
                        targets = rewards + (1 - dones) * args.gamma * max_next_q

                    predictions = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                    loss = loss_fn(predictions, targets)
                    td_abs = (predictions.detach() - targets).abs()

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.sqrt(
                        sum(
                            parameter.grad.detach().pow(2).sum()
                            for parameter in q_net.parameters()
                            if parameter.grad is not None
                        )
                    ).item()
                    optimizer.step()

                    step += 1
                    diagnostics["loss"].append(loss.item())
                    diagnostics["td_abs"].extend(td_abs.detach().cpu().tolist())
                    diagnostics["q_selected"].extend(predictions.detach().cpu().tolist())
                    diagnostics["q_max"].extend(q_net(states).detach().max(dim=1).values.cpu().tolist())
                    diagnostics["grad_norm"].append(grad_norm)

                action_step = completed_agent_step
                if should_update_target(action_step, args.target_update_freq):
                    target_net.load_state_dict(q_net.state_dict())
                    target_update_count += 1

                if action_step % args.diagnostics_every_agent_steps == 0:
                    def diag_mean(key):
                        values = diagnostics[key]
                        return float(np.mean(values)) if values else ""

                    diagnostics_writer.writerow({
                        "agent_step": action_step, "env_step": env_step, "train_step": step,
                        "episode": episode, "epsilon": f"{epsilon:.6f}",
                        "raw_episode_return": episode_reward,
                        "loss_mean": diag_mean("loss"), "td_error_abs_mean": diag_mean("td_abs"),
                        "td_error_abs_p95": float(np.percentile(diagnostics["td_abs"], 95)) if diagnostics["td_abs"] else "",
                        "q_selected_mean": diag_mean("q_selected"), "q_max_mean": diag_mean("q_max"),
                        "q_max_abs_max": float(np.max(np.abs(diagnostics["q_max"]))) if diagnostics["q_max"] else "",
                        "grad_norm_mean": diag_mean("grad_norm"),
                        "clip_neg1": clipped_reward_counts["-1"], "clip_zero": clipped_reward_counts["0"], "clip_pos1": clipped_reward_counts["+1"],
                        "life_loss_count": life_loss_count, "target_copy_count": target_update_count,
                        "action_counts": json.dumps(action_counts.tolist()),
                    })
                    diagnostics_file.flush()
                    action_counts.fill(0)
                    clipped_reward_counts = {"-1": 0, "0": 0, "+1": 0}
                    diagnostics = {"loss": [], "td_abs": [], "q_selected": [], "q_max": [], "grad_norm": []}

                state = next_state

            writer.writerow([episode, episode_reward, epsilon])
            f.flush()

            print(f"episode={episode} reward={episode_reward} epsilon={epsilon:.3f}", flush=True)

            if episode % args.checkpoint_every == 0 and episode > 0:
                q_net_path = outdir / f"q_net_ep{episode}.pt"
                checkpoint_path = outdir / f"checkpoint_ep{episode}.pt"
                torch.save(q_net.state_dict(), q_net_path)
                save_checkpoint(
                    checkpoint_path,
                    episode,
                    env_step,
                    action_step,
                    step,
                    target_update_count,
                    epsilon,
                    q_net,
                    target_net,
                    optimizer,
                    args,
                )
                append_checkpoint_manifest(
                    outdir,
                    episode,
                    env_step,
                    action_step,
                    step,
                    epsilon,
                    q_net_path.name,
                    checkpoint_path.name,
                )

            if (
                args.max_agent_steps_total is not None
                and action_step >= args.max_agent_steps_total
            ) or (
                args.max_raw_env_frames_total is not None
                and env_step >= args.max_raw_env_frames_total
            ):
                break

    q_net_final_path = outdir / "q_net_final.pt"
    checkpoint_final_path = outdir / "checkpoint_final.pt"
    torch.save(q_net.state_dict(), q_net_final_path)
    save_checkpoint(
        checkpoint_final_path,
        last_episode,
        env_step,
        action_step,
        step,
        target_update_count,
        epsilon if "epsilon" in locals() else args.epsilon_start,
        q_net,
        target_net,
        optimizer,
        args,
    )
    append_checkpoint_manifest(
        outdir,
        last_episode,
        env_step,
        action_step,
        step,
        epsilon if "epsilon" in locals() else args.epsilon_start,
        q_net_final_path.name,
        checkpoint_final_path.name,
    )
    diagnostics_file.close()
    env.close()


if __name__ == "__main__":
    main()
