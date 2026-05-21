import argparse
import csv
import random
from collections import deque
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gym.register_envs(ale_py)


def preprocess_frame(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


def make_initial_state(obs):
    frame = preprocess_frame(obs)
    frames = deque(maxlen=4)
    for _ in range(4):
        frames.append(frame)
    state = np.stack(frames, axis=0)
    return frames, state


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
    parser.add_argument("--target-update-freq", type=int, default=1_000)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--max-env-steps-total", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--debug-shapes", action="store_true")

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--start-env-step", type=int, default=0)
    parser.add_argument("--start-train-step", type=int, default=0)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id)
    num_actions = env.action_space.n

    q_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.RMSprop(q_net.parameters(), lr=args.lr, alpha=0.95, eps=0.01)
    loss_fn = nn.MSELoss()

    start_episode = args.start_episode
    env_step = args.start_env_step
    step = args.start_train_step

    if args.resume is not None:
        print(f"Loading checkpoint: {args.resume}", flush=True)
        checkpoint = torch.load(args.resume, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            q_net.load_state_dict(checkpoint["model_state_dict"])

            if "target_model_state_dict" in checkpoint:
                target_net.load_state_dict(checkpoint["target_model_state_dict"])
            else:
                target_net.load_state_dict(q_net.state_dict())

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_episode = checkpoint.get("episode", start_episode - 1) + 1
            env_step = checkpoint.get("env_step", env_step)
            step = checkpoint.get("train_step", step)

            print("Loaded full checkpoint.", flush=True)
        else:
            q_net.load_state_dict(checkpoint)
            target_net.load_state_dict(q_net.state_dict())
            print("Loaded weights-only checkpoint.", flush=True)

        print(f"Starting episode: {start_episode}", flush=True)
        print(f"Starting env_step: {env_step}", flush=True)
        print(f"Starting train_step: {step}", flush=True)

    replay_buffer = deque(maxlen=args.replay_size)
    printed_debug_shapes = False

    rewards_csv = outdir / "rewards.csv"
    csv_mode = "a" if args.resume is not None else "w"
    write_header = csv_mode == "w" or not rewards_csv.exists() or rewards_csv.stat().st_size == 0

    with open(rewards_csv, csv_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "reward", "epsilon"])

        for episode in range(start_episode, args.episodes):
            obs, info = env.reset()
            frames, state = make_initial_state(obs)
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    q_values = q_net(state_tensor)

                progress = min(env_step / args.epsilon_decay, 1.0)
                epsilon = args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax(dim=1).item()

                total_reward = 0.0
                next_obs = None

                for _ in range(args.frame_skip):
                    next_obs, reward, terminated, truncated, info = env.step(action)
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

                    done = terminated or truncated
                    if done:
                        break

                    if args.max_env_steps_total is not None and env_step >= args.max_env_steps_total:
                        done = True
                        break

                episode_steps += 1
                episode_reward += total_reward

                if args.max_steps_per_episode is not None and episode_steps >= args.max_steps_per_episode:
                    done = True

                next_frame = preprocess_frame(next_obs)
                frames.append(next_frame)
                next_state = np.stack(frames, axis=0)

                replay_buffer.append((state, action, total_reward, next_state, done))

                if len(replay_buffer) >= args.batch_size:
                    batch = random.sample(replay_buffer, args.batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                    actions = torch.tensor(actions, dtype=torch.long).to(device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(device)

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
        episode,
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
