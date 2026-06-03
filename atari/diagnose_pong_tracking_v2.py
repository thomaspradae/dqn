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


def detect_objects(processed):
    img = processed.copy()
    img[:18, :] = 0
    img[82:, :] = 0

    mask = img > 120

    # right paddle near right edge
    right_region = mask[:, 70:83]
    ys = np.where(right_region)[0]
    paddle_y = float(np.mean(ys)) if len(ys) else None

    # ball: small bright connected object away from paddles
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

    if not candidates:
        return paddle_y, None, None

    candidates.sort(reverse=True)
    _, ball_x, ball_y = candidates[0]
    return paddle_y, float(ball_x), float(ball_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--noop-max", type=int, default=30)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--deadzone", type=float, default=3.0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    env = gym.make("ALE/Pong-v5")
    meanings = env.unwrapped.get_action_meanings()
    print("action meanings:", meanings)

    q = QNetwork(env.action_space.n)
    q.load_state_dict(torch.load(args.model, map_location="cpu"))
    q.eval()

    action_counts = Counter()

    total_toward = 0
    improved = 0
    worsened = 0
    same = 0
    missed_detection = 0

    examples = []

    for ep in range(args.episodes):
        obs, info = reset_with_noops(env, args.noop_max)
        frames, state = make_state(obs)
        last_raw = obs
        prev_ball_x = None
        done = False

        while not done:
            current_proc = preprocess_frame(last_raw)
            paddle_y, ball_x, ball_y = detect_objects(current_proc)

            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = q(st)[0].numpy()

            action = int(np.argmax(qvals)) if np.random.rand() >= args.epsilon else env.action_space.sample()
            action_name = meanings[action]
            action_counts[action_name] += 1

            old_error = None
            ball_moving_right = False

            if paddle_y is not None and ball_y is not None and ball_x is not None and prev_ball_x is not None:
                ball_vx = ball_x - prev_ball_x
                ball_moving_right = ball_vx > 0.5 and ball_x > 35
                if ball_moving_right and abs(ball_y - paddle_y) >= args.deadzone:
                    old_error = abs(ball_y - paddle_y)

            prev_raw = last_raw
            for _ in range(args.frame_skip):
                prev_raw = last_raw
                obs2, reward, terminated, truncated, info = env.step(action)
                last_raw = obs2
                done = terminated or truncated
                if done:
                    break

            next_proc = preprocess_frame(last_raw, prev_raw)
            next_paddle_y, next_ball_x, next_ball_y = detect_objects(next_proc)

            frames.append(next_proc)
            state = np.stack(frames, axis=0)

            if old_error is not None:
                total_toward += 1
                if next_paddle_y is None or next_ball_y is None:
                    missed_detection += 1
                else:
                    new_error = abs(next_ball_y - next_paddle_y)
                    delta = old_error - new_error

                    if delta > 0.5:
                        improved += 1
                        outcome = "improved"
                    elif delta < -0.5:
                        worsened += 1
                        outcome = "worsened"
                    else:
                        same += 1
                        outcome = "same"

                    if len(examples) < 25:
                        examples.append({
                            "action": action_name,
                            "old_error": round(old_error, 2),
                            "new_error": round(new_error, 2),
                            "delta_error": round(delta, 2),
                            "outcome": outcome,
                        })

            if ball_x is not None:
                prev_ball_x = ball_x

    env.close()

    print()
    print("action_counts")
    for k, v in action_counts.most_common():
        print(f"{k:10s} {v}")

    print()
    print("toward-paddle tracking diagnostic")
    print(f"total_toward_decisions={total_toward}")
    print(f"improved={improved}")
    print(f"worsened={worsened}")
    print(f"same={same}")
    print(f"missed_detection={missed_detection}")

    denom = improved + worsened + same
    if denom:
        print(f"improve_rate={improved / denom:.3f}")
        print(f"worsen_rate={worsened / denom:.3f}")
        print(f"same_rate={same / denom:.3f}")

    print()
    print("examples")
    for e in examples:
        print(e)


if __name__ == "__main__":
    main()
