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


def get_noop_action(env):
    meanings = env.unwrapped.get_action_meanings()
    return meanings.index("NOOP") if "NOOP" in meanings else 0


def reset_with_noops(env, noop_max):
    obs, info = env.reset()
    noop = get_noop_action(env)
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


def detect_pong_objects(processed):
    """
    processed: 84x84 grayscale.
    Rough heuristic:
    - scoreboard/top border is ignored.
    - right paddle is near x 72-82.
    - ball is a small bright-ish object away from paddles.
    """
    img = processed.copy()

    # Ignore scoreboard / top border / bottom border
    img[:18, :] = 0
    img[82:, :] = 0

    # Threshold bright game objects
    mask = img > 120

    # Right paddle: bright vertical pixels near right side
    right_region = mask[:, 70:83]
    ys = np.where(right_region)[0]
    paddle_y = float(np.mean(ys)) if len(ys) else None

    # Ball: connected/small bright pixels not near paddles
    ball_mask = mask.copy()
    ball_mask[:, :12] = 0
    ball_mask[:, 68:] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ball_mask.astype(np.uint8), connectivity=8
    )

    candidates = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if 1 <= area <= 12 and 1 <= w <= 5 and 1 <= h <= 5:
            candidates.append((area, centroids[i][0], centroids[i][1]))

    if candidates:
        # choose largest small object
        candidates.sort(reverse=True)
        _, ball_x, ball_y = candidates[0]
        return paddle_y, float(ball_x), float(ball_y)

    return paddle_y, None, None


def action_direction(action_name):
    # For ALE Pong labels, empirically LEFT/LEFTFIRE and RIGHT/RIGHTFIRE are the two paddle moves.
    # We don't assume which is up yet; we report both raw correlations.
    if action_name in ("LEFT", "LEFTFIRE"):
        return "LEFT"
    if action_name in ("RIGHT", "RIGHTFIRE"):
        return "RIGHT"
    if action_name in ("NOOP", "FIRE"):
        return "STAY"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.0)
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
    rows = []

    for ep in range(args.episodes):
        obs, info = reset_with_noops(env, args.noop_max)
        frames, state = make_state(obs)
        last_raw = obs
        done = False

        while not done:
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = q(st)[0].numpy()

            if np.random.rand() < args.epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(qvals))

            name = meanings[action]
            action_counts[name] += 1

            prev_raw = last_raw
            total_reward = 0
            for _ in range(args.frame_skip):
                prev_raw = last_raw
                obs2, reward, terminated, truncated, info = env.step(action)
                last_raw = obs2
                total_reward += reward
                done = terminated or truncated
                if done:
                    break

            proc = preprocess_frame(last_raw, prev_raw)
            frames.append(proc)
            state = np.stack(frames, axis=0)

            paddle_y, ball_x, ball_y = detect_pong_objects(proc)
            if paddle_y is not None and ball_y is not None and ball_x is not None:
                # Only when ball is on right half / approaching area enough to matter
                if ball_x > 35:
                    rows.append({
                        "paddle_y": paddle_y,
                        "ball_y": ball_y,
                        "dy": ball_y - paddle_y,
                        "action": name,
                        "dir": action_direction(name),
                    })

    env.close()

    print()
    print("action_counts")
    for k, v in action_counts.most_common():
        print(f"{k:10s} {v}")

    print()
    print("detected decisions near right side:", len(rows))
    if not rows:
        return

    # We do not know a priori whether LEFT means up or down in this ALE label mapping,
    # so compute both interpretations.
    useful = [r for r in rows if abs(r["dy"]) >= 3]
    print("useful decisions abs(ball_y - paddle_y) >= 3:", len(useful))

    def score(mapping):
        correct = 0
        wrong = 0
        stay = 0
        for r in useful:
            desired = "DOWN" if r["dy"] > 0 else "UP"
            d = r["dir"]
            if d == "STAY":
                stay += 1
                continue
            interpreted = mapping.get(d)
            if interpreted == desired:
                correct += 1
            else:
                wrong += 1
        total = correct + wrong + stay
        return correct, wrong, stay, total

    mappings = {
        "LEFT=UP RIGHT=DOWN": {"LEFT": "UP", "RIGHT": "DOWN"},
        "LEFT=DOWN RIGHT=UP": {"LEFT": "DOWN", "RIGHT": "UP"},
    }

    for name, mapping in mappings.items():
        correct, wrong, stay, total = score(mapping)
        acc = correct / total if total else 0
        print()
        print(name)
        print(f"correct={correct} wrong={wrong} stay={stay} total={total}")
        print(f"tracking_accuracy_including_stay={acc:.3f}")

    print()
    print("Sample rows:")
    for r in rows[:20]:
        print(r)


if __name__ == "__main__":
    main()
