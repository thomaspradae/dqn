#!/bin/bash
#SBATCH --job-name=pong-nature-smoke
#SBATCH --partition=office
#SBATCH --nodelist=ofi1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pong-nature-smoke-%j.out
#SBATCH --error=logs/pong-nature-smoke-%j.err

set -euo pipefail

cd /home/uace/dqn/atari
source /home/uace/dqn/.venv/bin/activate

mkdir -p logs runs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "SBATCH START $(date -Is)"
echo "HOST=$(hostname)"
echo "PYTHON=$(which python)"

python train_nature.py \
  --env-id ALE/Pong-v5 \
  --episodes 100 \
  --max-raw-env-frames-total 60000 \
  --outdir runs/pong_nature_smoke_slurm \
  --debug-shapes
