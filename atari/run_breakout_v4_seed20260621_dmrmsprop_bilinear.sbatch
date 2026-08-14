#!/bin/bash
#SBATCH --job-name=breakout-v4-dmrms-bilinear
#SBATCH --partition=office
#SBATCH --nodelist=ofi2
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home/uace/dqn/atari/logs/breakout_v4_seed20260621_dmrmsprop_bilinear.out
#SBATCH --error=/home/uace/dqn/atari/logs/breakout_v4_seed20260621_dmrmsprop_bilinear.err

set -euo pipefail
cd /home/uace/dqn/atari
source /home/uace/dqn/.venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "SBATCH START $(date -Is)"
python train_nature.py \
  --env-id ALE/Breakout-v5 \
  --run-dir /home/uace/dqn/atari/runs/breakout_v4_seed20260621_dmrmsprop_bilinear \
  --max-agent-steps-total 50000000 \
  --seed 20260621 \
  --optimizer deepmind_rmsprop \
  --resize-interpolation bilinear \
  --frame-skip 4 \
  --train-freq 4 \
  --learning-starts 50000 \
  --epsilon-decay 1000000 \
  --target-update-freq 10000 \
  --replay-size 1000000 \
  --checkpoint-every 100 \
  --diagnostics-every-agent-steps 50000 \
  --threads 8 \
  --life-loss-terminal
echo "SBATCH END $(date -Is)"
