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


def preprocess_frame(obs, previous_obs=None):
    if previous_obs is not None:
        obs = np.maximum(obs, previous_obs)

    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


def make_initial_state(obs):
    frame = preprocess_frame(obs)
    frames = deque(maxlen=4)
    for _ in range(4):
        frames.append(frame)
    return frames, np.stack(frames, axis=0)


def get_noop_action(env):
    get_action_meanings = getattr(env.unwrapped, "get_action_meanings", None)
    if get_action_meanings is None:
        return 0

    action_meanings = get_action_meanings()
    if "NOOP" in action_meanings:
        return action_meanings.index("NOOP")

    return 0


def reset_with_noops(env, noop_max):
    obs, info = env.reset()

    if noop_max <= 0:
        return obs, info

    noop_action = get_noop_action(env)
    for _ in range(random.randint(1, noop_max)):
        obs, _, terminated, truncated, info = env.step(noop_action)
        if terminated or truncated:
            obs, info = env.reset()

    return obs, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--model", type=str, default="runs/pong_cpu_3day/q_net_final.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.01)
    args = parser.parse_args()

    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")
    if args.noop_max < 0:
        parser.error("--noop-max must be >= 0")

    torch.set_num_threads(args.threads)

    env = gym.make(args.env_id, render_mode="human", frameskip=1, repeat_action_probability=0.0)
    q_net = QNetwork(env.action_space.n)

    checkpoint = torch.load(args.model, map_location="cpu")
    q_net.load_state_dict(checkpoint)
    q_net.eval()

    print(f"Loaded model: {args.model}", flush=True)

    for episode in range(args.episodes):
        obs, info = reset_with_noops(env, args.noop_max)
        frames, state = make_initial_state(obs)
        last_raw_obs = obs

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
            previous_raw_obs = last_raw_obs

            for _ in range(args.frame_skip):
                previous_raw_obs = last_raw_obs
                next_obs, reward, terminated, truncated, info = env.step(action)
                last_raw_obs = next_obs
                total_reward += reward
                env.render()
                time.sleep(args.sleep)

                done = terminated or truncated
                if done:
                    break

            episode_reward += total_reward

            next_frame = preprocess_frame(last_raw_obs, previous_raw_obs)
            frames.append(next_frame)
            state = np.stack(frames, axis=0)

        print(f"episode={episode} reward={episode_reward}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
