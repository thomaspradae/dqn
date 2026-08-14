#!/bin/bash
#SBATCH --job-name=dqn-pong-3d
#SBATCH --output=/home/uace/dqn/atari/logs/dqn-pong-3d-%j.out
#SBATCH --error=/home/uace/dqn/atari/logs/dqn-pong-3d-%j.err
#SBATCH --partition=office
#SBATCH --nodelist=ofi1
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G

set -euo pipefail

cd /home/uace/dqn/atari

source /home/uace/dqn/.venv/bin/activate

mkdir -p logs runs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python train.py \
  --episodes 100000 \
  --batch-size 32 \
  --replay-size 50000 \
  --frame-skip 4 \
  --threads 8 \
  --max-env-steps-total 10000000 \
  --checkpoint-every 100 \
  --outdir runs/pong_cpu_3day
