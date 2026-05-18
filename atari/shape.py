from collections import deque

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch

from network import QNetwork

gym.register_envs(ale_py)


def preprocess_frame(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


def main():
    env = gym.make("ALE/Pong-v5")

    num_actions = env.action_space.n
    q_net = QNetwork(num_actions)

    obs, info = env.reset()

    print("raw obs shape:", obs.shape)
    print("num actions:", num_actions)

    frame = preprocess_frame(obs)
    print("processed frame shape:", frame.shape)

    frames = deque(maxlen=4)

    for _ in range(4):
        frames.append(frame)

    state = np.stack(frames, axis=0)
    print("state shape:", state.shape)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    print("state tensor shape:", state_tensor.shape)

    with torch.no_grad():
        q_values = q_net(state_tensor)

    print("q_values shape:", q_values.shape)
    print("q_values:", q_values)

    action = q_values.argmax(dim=1).item()
    print("chosen action:", action)

    env.close()


if __name__ == "__main__":
    main()
