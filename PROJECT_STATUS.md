# DQN CartPole Status

## Completed

- Implemented a basic Deep Q-Network training loop for `CartPole-v1`.
- Added replay buffer sampling.
- Added target network updates.
- Added checkpoint saving during training.
- Logged per-episode rewards to `runs/slurm-9/rewards.csv`.
- Trained a CartPole agent on the cluster.
- Saved checkpoints in `runs/slurm-9/`.
- Evaluated the learned policy and observed perfect `500/500` reward episodes.
- Generated a learning curve artifact:
  - Plot: `runs/slurm-9/learning_curve.png`
  - Commit-friendly plot copy: `artifacts/cartpole_slurm9_learning_curve.png`
  - Plot script: `scripts/plot_rewards.py`

## Current Result

- Training run: `runs/slurm-9/rewards.csv`
- Episodes: 1000
- Best training reward: 500
- First training episode with reward 500: episode 188
- Number of training episodes with reward 500: 402
- Last-100 training reward average: 343.56

The training curve is noisy because `train.py` currently uses fixed epsilon-greedy exploration. Greedy evaluation can score perfectly even while training rewards fluctuate, because training is still intentionally taking random actions with `epsilon=0.1`.

## Missing

- Add epsilon decay to `train.py`.
- Log epsilon per episode, not only reward.
- Save evaluation results to a CSV file.
- Add a README explaining the DQN implementation, how to train, how to evaluate, and how to reproduce the plot.
- Add requirements or environment setup instructions.
- Add a cleaner CLI for selecting environment names and model output directories.
- Add random seed support for reproducible runs.
- Consider plotting evaluation rewards separately from training rewards.

## Recommended Next Step

Before jumping straight to Pong, make the CartPole version cleaner:

- Add epsilon decay.
- Run one clean CartPole training job.
- Save a new learning curve that should climb and stay near 500 more cleanly.
- Write the README while the experiment is fresh.

After that, try one intermediate environment before Pong:

- `Acrobot-v1`: no pixel preprocessing, no extra Box2D dependency, but harder than CartPole.
- `LunarLander-v3`: a better demonstration if Box2D is installed, but it may require extra dependencies.

Pong is the right bigger milestone, but it is a different project shape: convolutional network, frame preprocessing, frame stacking, larger replay buffer, longer training, and likely GPU use.

## Pong Prep Checklist

- Add Atari/Gymnasium environment support.
- Add image preprocessing: grayscale, resize, crop if needed.
- Stack frames.
- Replace the MLP with a convolutional Q-network.
- Use a larger replay buffer.
- Add epsilon decay schedule.
- Add periodic evaluation without exploration.
- Save videos or GIFs of trained policy rollouts.
- Run long jobs through Slurm.
