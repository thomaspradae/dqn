#!/bin/bash
#SBATCH --job-name=dqn-pong-30m
#SBATCH --partition=office
#SBATCH --nodelist=uaceofi1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/dqn-pong-30m-%j.out
#SBATCH --error=logs/dqn-pong-30m-%j.err

set -euo pipefail

cd /home/uace/dqn/atari
source /home/uace/dqn/.venv/bin/activate

mkdir -p logs runs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "SBATCH START $(date -Is)"
echo "PYTHON=$(which python)"

python train_serious.py \
  --episodes 100000 \
  --batch-size 32 \
  --replay-size 50000 \
  --frame-skip 4 \
  --threads 8 \
  --max-env-steps-total 30000000 \
  --checkpoint-every 100 \
  --outdir runs/pong_cpu_30m \
  --resume runs/pong_cpu_30m/q_net_final.pt \
  --start-episode 4305 \
  --start-env-step 10000000 \
  --start-train-step 1414414
