import argparse
from collections import deque, Counter

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
    noop = meanings.index("NOOP") if "NOOP" in meanings else 0
    for _ in range(np.random.randint(1, noop_max + 1)):
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
    ap.add_argument("--noop-max", type=int, default=30)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    env = gym.make("ALE/Pong-v5")
    meanings = env.unwrapped.get_action_meanings()
    print("action meanings:", meanings)

    q = QNetwork(env.action_space.n)
    q.load_state_dict(torch.load(args.model, map_location="cpu"))
    q.eval()

    action_counts = Counter()
    q_records = []
    margins = []

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

            order = np.argsort(qvals)[::-1]
            best = int(order[0])
            second = int(order[1])
            margin = float(qvals[best] - qvals[second])

            action_counts[meanings[best]] += 1
            margins.append(margin)
            q_records.append(qvals)

            prev_raw = last_raw
            for _ in range(args.frame_skip):
                prev_raw = last_raw
                obs2, reward, terminated, truncated, info = env.step(best)
                last_raw = obs2
                ep_reward += reward
                done = terminated or truncated
                if done:
                    break

            proc = preprocess_frame(last_raw, prev_raw)
            frames.append(proc)
            state = np.stack(frames, axis=0)

        rewards.append(ep_reward)

    env.close()

    q_arr = np.array(q_records)
    margins = np.array(margins)

    print()
    print("eval rewards")
    print(f"episodes={len(rewards)} mean={np.mean(rewards):.2f} std={np.std(rewards):.2f} min={np.min(rewards):.2f} max={np.max(rewards):.2f}")

    print()
    print("greedy action counts")
    for k, v in action_counts.most_common():
        print(f"{k:10s} {v}")

    print()
    print("q-value summary by action")
    for i, name in enumerate(meanings):
        vals = q_arr[:, i]
        print(f"{name:10s} mean={vals.mean():8.3f} std={vals.std():8.3f} min={vals.min():8.3f} max={vals.max():8.3f}")

    print()
    print("best-vs-second margin")
    print(f"mean={margins.mean():.4f}")
    print(f"median={np.median(margins):.4f}")
    print(f"p10={np.percentile(margins, 10):.4f}")
    print(f"p90={np.percentile(margins, 90):.4f}")

    print()
    print("interpretation hint")
    print("If FIRE has the highest mean or dominates greedy counts, the policy may be overvaluing FIRE.")
    print("If margins are tiny, the policy is indecisive: actions have very similar Q-values.")
    print("If Q-values are huge or all nearly equal, that suggests value-scale / learning instability.")


if __name__ == '__main__':
    main()
