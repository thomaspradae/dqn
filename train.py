import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    if args.outdir is None:
        job_id = os.environ.get("SLURM_JOB_ID")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        if job_id is not None:
            args.outdir = f"runs/slurm-{job_id}"
        else:
            args.outdir = f"runs/local-{stamp}"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = gym.make("CartPole-v1")  # no render_mode on cluster

    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=args.replay_size)
    step = 0

    rewards_csv = outdir / "rewards.csv"

    with open(rewards_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])

        for episode in range(args.episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)

                with torch.no_grad():
                    q_values = q_net(obs_tensor)

                epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(-step / args.epsilon_decay)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax().item()

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                replay_buffer.append((obs, action, next_obs, reward, done))

                if len(replay_buffer) >= args.batch_size:
                    batch = random.sample(replay_buffer, args.batch_size)
                    states, actions, next_states, rewards, dones = zip(*batch)

                    states = torch.tensor(np.array(states), dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.long)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                    dones = torch.tensor(dones, dtype=torch.float32)

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

                obs = next_obs

            writer.writerow([episode, episode_reward])
            f.flush()

            print(f"episode={episode} reward={episode_reward}", flush=True)

            if episode % args.checkpoint_every == 0 and episode > 0:
                torch.save(q_net.state_dict(), outdir / f"q_net_ep{episode}.pt")

    torch.save(q_net.state_dict(), outdir / "q_net_final.pt")
    env.close()


if __name__ == "__main__":
    main()
