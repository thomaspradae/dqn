#!/bin/bash
#SBATCH --job-name=pong-paperfix-50m
#SBATCH --partition=office
#SBATCH --nodelist=uaceofi1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pong-paperfix-50m-%j.out
#SBATCH --error=logs/pong-paperfix-50m-%j.err

set -euo pipefail

cd /home/uace/dqn/atari
source /home/uace/dqn/.venv/bin/activate

mkdir -p logs runs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "SBATCH START $(date -Is)"
echo "RUN=pong_nature_paperfix_50m"
echo "HOST=$(hostname)"
echo "PYTHON=$(which python)"
free -h

python train_nature.py \
  --env-id ALE/Pong-v5 \
  --episodes 100000 \
  --max-env-steps-total 50000000 \
  --outdir runs/pong_nature_paperfix_50m \
  --checkpoint-every 100

free -h
echo "SBATCH END $(date -Is)"
