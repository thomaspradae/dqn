import argparse
import csv
import random
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from network import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gym.register_envs(ale_py)


def preprocess_frame(obs, previous_obs=None):
    if previous_obs is not None:
        obs = np.maximum(obs, previous_obs)

    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
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


def get_noop_action(env):
    get_action_meanings = getattr(env.unwrapped, "get_action_meanings", None)
    if get_action_meanings is None:
        return 0

    action_meanings = get_action_meanings()
    if "NOOP" in action_meanings:
        return action_meanings.index("NOOP")

    return 0


def reset_with_noops(env, noop_max, max_noop_steps=None):
    obs, info = env.reset()

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


def save_checkpoint(path, episode, env_step, train_step, q_net, target_net, optimizer, args):
    torch.save(
        {
            "episode": episode,
            "env_step": env_step,
            "train_step": train_step,
            "model_state_dict": q_net.state_dict(),
            "target_model_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="runs/atari")
    parser.add_argument("--lr", type=float, default=2.5e-4)
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
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--max-env-steps-total", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--debug-shapes", action="store_true")
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
    if args.max_env_steps_total is not None and args.max_env_steps_total < 1:
        parser.error("--max-env-steps-total must be >= 1")
    if args.resume is not None:
        parser.error("--resume is disabled because replay memory is not checkpointed")
    if args.start_episode != 0 or args.start_env_step != 0 or args.start_train_step != 0:
        parser.error("--start-* counters are disabled because replay memory is not checkpointed")

    torch.set_num_threads(args.threads)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, frameskip=1, repeat_action_probability=0.0)
    num_actions = env.action_space.n

    q_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.RMSprop(
        q_net.parameters(),
        lr=args.lr,
        alpha=0.95,
        eps=0.01,
        momentum=0.95,
    )
    loss_fn = clipped_td_error_loss

    start_episode = 0
    env_step = 0
    step = 0
    action_step = 0

    replay_buffer = ReplayMemory(args.replay_size)
    printed_debug_shapes = False

    rewards_csv = outdir / "rewards.csv"
    csv_mode = "w"
    write_header = True

    last_episode = start_episode - 1

    with open(rewards_csv, csv_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward", "epsilon"])

        for episode in range(start_episode, args.episodes):
            last_episode = episode
            max_noop_steps = None
            if args.max_env_steps_total is not None:
                max_noop_steps = max(args.max_env_steps_total - env_step, 0)

            obs, info, noop_steps = reset_with_noops(
                env,
                args.noop_max,
                max_noop_steps=max_noop_steps,
            )
            env_step += noop_steps

            if args.max_env_steps_total is not None and env_step >= args.max_env_steps_total:
                break

            state = replay_buffer.start_episode(preprocess_frame(obs))
            last_raw_obs = obs
            lives = get_lives(env)
            episode_done = False
            episode_reward = 0.0
            episode_steps = 0
            epsilon = args.epsilon_start

            while not episode_done:
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    q_values = q_net(state_tensor)

                progress = min(env_step / args.epsilon_decay, 1.0)
                epsilon = args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax(dim=1).item()

                total_reward = 0.0
                previous_raw_obs = last_raw_obs
                env_done = False
                forced_done = False
                life_lost = False

                for _ in range(args.frame_skip):
                    previous_raw_obs = last_raw_obs
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    last_raw_obs = next_obs
                    env_step += 1

                    clipped_reward = max(-1.0, min(1.0, reward))
                    total_reward += clipped_reward

                    if env_step % 1000 == 0:
                        print(
                            f"env_step={env_step} train_step={step} "
                            f"episode={episode} reward={episode_reward + total_reward} "
                            f"epsilon={epsilon:.3f}",
                            flush=True,
                        )

                    env_done = terminated or truncated

                    current_lives = get_lives(env)
                    if args.life_loss_terminal and lives > 0 and current_lives < lives:
                        life_lost = True
                    lives = current_lives

                    if args.max_env_steps_total is not None and env_step >= args.max_env_steps_total:
                        forced_done = True

                    if env_done or forced_done or life_lost:
                        break

                episode_steps += 1
                action_step += 1
                episode_reward += total_reward
                episode_done = env_done or forced_done
                transition_terminal = episode_done or life_lost

                if args.max_steps_per_episode is not None and episode_steps >= args.max_steps_per_episode:
                    episode_done = True
                    transition_terminal = True

                next_frame = preprocess_frame(last_raw_obs, previous_raw_obs)
                next_state = replay_buffer.append(
                    action,
                    total_reward,
                    next_frame,
                    transition_terminal,
                )
                if life_lost and not episode_done:
                    next_state = replay_buffer.start_episode(next_frame)

                if (
                    env_step >= args.learning_starts
                    and len(replay_buffer) >= args.batch_size
                    and action_step % args.train_freq == 0
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

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step += 1

                    if step % args.target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())

                state = next_state

            writer.writerow([episode, episode_reward, epsilon])
            f.flush()

            print(f"episode={episode} reward={episode_reward} epsilon={epsilon:.3f}", flush=True)

            if episode % args.checkpoint_every == 0 and episode > 0:
                torch.save(q_net.state_dict(), outdir / f"q_net_ep{episode}.pt")
                save_checkpoint(
                    outdir / f"checkpoint_ep{episode}.pt",
                    episode,
                    env_step,
                    step,
                    q_net,
                    target_net,
                    optimizer,
                    args,
                )

            if args.max_env_steps_total is not None and env_step >= args.max_env_steps_total:
                break

    torch.save(q_net.state_dict(), outdir / "q_net_final.pt")
    save_checkpoint(
        outdir / "checkpoint_final.pt",
        last_episode,
        env_step,
        step,
        q_net,
        target_net,
        optimizer,
        args,
    )
    env.close()


if __name__ == "__main__":
    main()
