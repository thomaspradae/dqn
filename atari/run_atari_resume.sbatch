#!/bin/bash
#SBATCH --job-name=dqn-pong-resume
#SBATCH --partition=office
#SBATCH --nodelist=ofi1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/dqn-pong-resume-%j.out
#SBATCH --error=logs/dqn-pong-resume-%j.err

set -euo pipefail

cd /home/uace/dqn/atari

source /home/uace/dqn/.venv/bin/activate

mkdir -p logs runs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "SBATCH START $(date -Is)"
echo "PYTHON=$(which python)"
echo "RESUME_CKPT=runs/pong_cpu_3day/q_net_ep2500.pt"
echo "START_ENV=4345000"

python train_resume.py \
  --episodes 100000 \
  --batch-size 32 \
  --replay-size 50000 \
  --frame-skip 4 \
  --threads 8 \
  --max-env-steps-total 10000000 \
  --checkpoint-every 100 \
  --outdir runs/pong_cpu_3day \
  --resume "runs/pong_cpu_3day/q_net_ep2500.pt" \
  --start-env-step "4345000"
