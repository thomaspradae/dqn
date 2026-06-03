import argparse
from collections import deque
import random

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
    y = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
    return cv2.resize(y, (84, 84), interpolation=cv2.INTER_AREA)

def reset_with_noops(env, noop_max):
    obs, info = env.reset()
    meanings = env.unwrapped.get_action_meanings()
    noop = meanings.index("NOOP")
    for _ in range(random.randint(1, noop_max)):
        obs, _, terminated, truncated, info = env.step(noop)
        if terminated or truncated:
            obs, info = env.reset()
    return obs, info

def make_state(obs):
    f = preprocess_frame(obs)
    frames = deque(maxlen=4)
    for _ in range(4):
        frames.append(f)
    return frames, np.stack(frames, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--noop-max", type=int, default=30)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    env = gym.make("ALE/Pong-v5")
    meanings = env.unwrapped.get_action_meanings()
    print("action meanings:", meanings)

    # For Pong: keep movement actions, remove NOOP/FIRE.
    allowed_names = {"RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"}
    allowed = [i for i, name in enumerate(meanings) if name in allowed_names]
    print("allowed:", [(i, meanings[i]) for i in allowed])

    q = QNetwork(env.action_space.n)
    q.load_state_dict(torch.load(args.model, map_location="cpu"))
    q.eval()

    rewards = []

    for ep in range(args.episodes):
        obs, info = reset_with_noops(env, args.noop_max)
        frames, state = make_state(obs)
        last_raw = obs
        done = False
        ep_reward = 0.0

        while not done:
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = q(st)[0].numpy()

            if random.random() < args.epsilon:
                action = random.choice(allowed)
            else:
                action = max(allowed, key=lambda a: qvals[a])

            prev_raw = last_raw
            for _ in range(args.frame_skip):
                prev_raw = last_raw
                obs2, reward, terminated, truncated, info = env.step(action)
                last_raw = obs2
                ep_reward += reward
                done = terminated or truncated
                if done:
                    break

            proc = preprocess_frame(last_raw, prev_raw)
            frames.append(proc)
            state = np.stack(frames, axis=0)

        rewards.append(ep_reward)
        print(f"episode={ep} reward={ep_reward}", flush=True)

    arr = np.array(rewards)
    print()
    print("Summary")
    print(f"episodes={len(arr)}")
    print(f"mean={arr.mean():.2f}")
    print(f"std={arr.std():.2f}")
    print(f"min={arr.min():.2f}")
    print(f"max={arr.max():.2f}")

if __name__ == "__main__":
    main()
