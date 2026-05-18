import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("runs/pong_cpu_800/rewards.csv")

df = pd.read_csv(
    path,
    header=None,
    names=["episode", "reward", "epsilon"],
)

df["rolling_50"] = df["reward"].rolling(50).mean()

out = path.parent / "rewards.png"

plt.figure()
plt.plot(df["episode"], df["reward"], alpha=0.35, label="episode reward")
plt.plot(df["episode"], df["rolling_50"], label="rolling 50")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Pong CPU run")
plt.legend()
plt.savefig(out, dpi=150)

print(df.head())
print(df.tail())
print(f"saved {out}")