import argparse
import random
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
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


def make_initial_state(obs):
    frame = preprocess_frame(obs)
    frames = deque(maxlen=4)
    for _ in range(4):
        frames.append(frame)
    return frames, np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--model", default="runs/pong_cpu_3day/q_net_final.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="pong_eval.mp4")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    env = gym.make(args.env_id, render_mode="rgb_array")
    q_net = QNetwork(env.action_space.n)
    q_net.load_state_dict(torch.load(args.model, map_location="cpu"))
    q_net.eval()

    writer = None
    rewards = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        frames, state = make_initial_state(obs)
        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = q_net(state_tensor)

            action = env.action_space.sample() if random.random() < args.epsilon else q_values.argmax(dim=1).item()

            total_reward = 0.0
            next_obs = None

            for _ in range(args.frame_skip):
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if writer is None:
                    h, w, _ = frame_bgr.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.out, fourcc, 60, (w, h))

                writer.write(frame_bgr)

                done = terminated or truncated
                if done:
                    break

            episode_reward += total_reward

            if next_obs is not None:
                next_frame = preprocess_frame(next_obs)
                frames.append(next_frame)
                state = np.stack(frames, axis=0)

        rewards.append(episode_reward)
        print(f"episode={episode} reward={episode_reward}", flush=True)

    if writer is not None:
        writer.release()
    env.close()

    arr = np.array(rewards, dtype=np.float32)
    print(f"saved video: {args.out}")
    print(f"mean={arr.mean():.2f} min={arr.min():.2f} max={arr.max():.2f}")


if __name__ == "__main__":
    main()
