#!/bin/bash
#SBATCH --job-name=pong-nature-50m
#SBATCH --partition=office
#SBATCH --nodelist=ofi1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pong-nature-50m-%j.out
#SBATCH --error=logs/pong-nature-50m-%j.err

set -euo pipefail

cd /home/uace/dqn/atari
source /home/uace/dqn/.venv/bin/activate

mkdir -p logs runs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "SBATCH START $(date -Is)"
echo "HOST=$(hostname)"
echo "PYTHON=$(which python)"
free -h

python train_nature.py \
  --env-id ALE/Pong-v5 \
  --episodes 100000 \
  --max-raw-env-frames-total 50000000 \
  --outdir runs/pong_nature_50m \
  --checkpoint-every 100

free -h
echo "SBATCH END $(date -Is)"
