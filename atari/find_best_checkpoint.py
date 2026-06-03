import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--window", type=int, default=100)
parser.add_argument("--checkpoint-every", type=int, default=100)
args = parser.parse_args()

run = Path(args.run_dir)
rewards_path = run / "rewards.csv"

rewards = []
with rewards_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        rewards.append((int(row["episode"]), float(row["reward"])))

if len(rewards) < args.window:
    raise SystemExit(f"Not enough rewards: {len(rewards)} < {args.window}")

best = None

for i in range(args.window - 1, len(rewards)):
    ep = rewards[i][0]
    avg = sum(r for _, r in rewards[i - args.window + 1:i + 1]) / args.window

    if best is None or avg > best["avg"]:
        best = {"episode": ep, "avg": avg}

ep = best["episode"]
rounded = (ep // args.checkpoint_every) * args.checkpoint_every

q_net = run / f"q_net_ep{rounded}.pt"
ckpt = run / f"checkpoint_ep{rounded}.pt"

print(f"run_dir: {run}")
print(f"episodes: {len(rewards)}")
print(f"best_episode: {ep}")
print(f"best_last{args.window}: {best['avg']:.4f}")
print(f"nearest_checkpoint_episode: {rounded}")
print(f"q_net: {q_net} exists={q_net.exists()}")
print(f"checkpoint: {ckpt} exists={ckpt.exists()}")
