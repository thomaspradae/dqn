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


def preprocess_frame(obs, previous_obs=None):
    if previous_obs is not None:
        obs = np.maximum(obs, previous_obs)

    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


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


def get_lives(env):
    ale = getattr(env.unwrapped, "ale", None)
    if ale is None:
        return None
    return ale.lives()


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
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--model", default="runs/pong_cpu_3day/q_net_final.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="pong_eval.mp4")
    parser.add_argument("--trace-lives", action="store_true")
    args = parser.parse_args()

    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")
    if args.noop_max < 0:
        parser.error("--noop-max must be >= 0")

    torch.set_num_threads(args.threads)

    env = gym.make(args.env_id, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
    q_net = QNetwork(env.action_space.n)
    q_net.load_state_dict(torch.load(args.model, map_location="cpu"))
    q_net.eval()

    writer = None
    rewards = []

    for episode in range(args.episodes):
        obs, info = reset_with_noops(env, args.noop_max)
        frames, state = make_initial_state(obs)
        last_raw_obs = obs
        lives = get_lives(env)
        if args.trace_lives:
            print(f"episode={episode} reset lives={lives}", flush=True)
        done = False
        episode_reward = 0.0
        raw_step = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = q_net(state_tensor)

            action = env.action_space.sample() if random.random() < args.epsilon else q_values.argmax(dim=1).item()

            total_reward = 0.0
            previous_raw_obs = last_raw_obs

            for _ in range(args.frame_skip):
                previous_raw_obs = last_raw_obs
                next_obs, reward, terminated, truncated, info = env.step(action)
                last_raw_obs = next_obs
                total_reward += reward
                raw_step += 1

                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if writer is None:
                    h, w, _ = frame_bgr.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.out, fourcc, 60, (w, h))

                writer.write(frame_bgr)

                done = terminated or truncated
                current_lives = get_lives(env)
                if args.trace_lives and current_lives != lives:
                    print(
                        f"episode={episode} raw_step={raw_step} life_change {lives}->{current_lives} "
                        f"action={action} reward={reward} done={done}",
                        flush=True,
                    )
                lives = current_lives
                if done:
                    break

            episode_reward += total_reward

            next_frame = preprocess_frame(last_raw_obs, previous_raw_obs)
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
