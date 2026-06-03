import argparse
from collections import deque
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)


def preprocess_frame(obs, previous_obs=None):
    if previous_obs is not None:
        obs = np.maximum(obs, previous_obs)

    y = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
    resized = cv2.resize(y, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


def get_noop_action(env):
    meanings = env.unwrapped.get_action_meanings()
    return meanings.index("NOOP") if "NOOP" in meanings else 0


def reset_with_noops(env, noop_max):
    obs, info = env.reset()
    noop = get_noop_action(env)

    for _ in range(noop_max):
        obs, _, terminated, truncated, info = env.step(noop)
        if terminated or truncated:
            obs, info = env.reset()

    return obs, info


def save_gray(path, img):
    cv2.imwrite(str(path), img)


def save_rgb(path, img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--outdir", default="input_inspect")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array")
    print("env:", args.env_id)
    print("action meanings:", env.unwrapped.get_action_meanings())

    obs, info = reset_with_noops(env, args.noop_max)
    frames = deque(maxlen=4)

    first = preprocess_frame(obs)
    for _ in range(4):
        frames.append(first)

    last_raw = obs

    for t in range(args.steps):
        action = env.action_space.sample()

        previous_raw = last_raw
        for _ in range(args.frame_skip):
            previous_raw = last_raw
            next_obs, reward, terminated, truncated, info = env.step(action)
            last_raw = next_obs
            if terminated or truncated:
                break

        maxed_rgb = np.maximum(last_raw, previous_raw)
        processed = preprocess_frame(last_raw, previous_raw)
        frames.append(processed)
        stack = np.concatenate(list(frames), axis=1)

        save_rgb(out / f"{t:03d}_raw_rgb.png", last_raw)
        save_rgb(out / f"{t:03d}_maxed_rgb.png", maxed_rgb)
        save_gray(out / f"{t:03d}_processed_84.png", processed)
        save_gray(out / f"{t:03d}_stack_4x84x84.png", stack)

        if terminated or truncated:
            break

    env.close()
    print("saved to:", out)


if __name__ == "__main__":
    main()
