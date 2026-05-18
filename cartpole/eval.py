import torch
import gymnasium as gym
from network import QNetwork

def eval(checkpoint_path, episodes=10):
    env = gym.make("CartPole-v1", render_mode="human")
    q_net = QNetwork()
    q_net.load_state_dict(torch.load(checkpoint_path))
    q_net.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0
        while not done:
            with torch.no_grad():
                action = torch.tensor(obs, dtype=torch.float32)
                action = q_net(action).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        print(f"episode={ep} reward={total}")
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    
    eval(args.checkpoint, args.episodes)