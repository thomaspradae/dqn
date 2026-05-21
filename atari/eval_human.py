import argparse
import random
import time
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch

from network import QNetwork

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
    return frames, np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--model", type=str, default="runs/pong_cpu_3day/q_net_final.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.01)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    env = gym.make(args.env_id, render_mode="human")
    q_net = QNetwork(env.action_space.n)

    checkpoint = torch.load(args.model, map_location="cpu")
    q_net.load_state_dict(checkpoint)
    q_net.eval()

    print(f"Loaded model: {args.model}", flush=True)

    for episode in range(args.episodes):
        obs, info = env.reset()
        frames, state = make_initial_state(obs)

        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = q_net(state_tensor)

            if random.random() < args.epsilon:
                action = env.action_space.sample()
            else:
                action = q_values.argmax(dim=1).item()

            total_reward = 0.0
            next_obs = None

            for _ in range(args.frame_skip):
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                env.render()
                time.sleep(args.sleep)

                done = terminated or truncated
                if done:
                    break

            episode_reward += total_reward

            if next_obs is not None:
                next_frame = preprocess_frame(next_obs)
                frames.append(next_frame)
                state = np.stack(frames, axis=0)

        print(f"episode={episode} reward={episode_reward}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
