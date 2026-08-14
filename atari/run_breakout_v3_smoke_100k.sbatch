#!/bin/bash
#SBATCH --job-name=breakout-v3-smoke
#SBATCH --partition=office
#SBATCH --nodelist=ofi1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=10:00:00
#SBATCH --output=/home/uace/dqn/atari/logs/breakout_v3_smoke_seed20260621_rewardclip_dmdec100k_retry2.out
#SBATCH --error=/home/uace/dqn/atari/logs/breakout_v3_smoke_seed20260621_rewardclip_dmdec100k_retry2.err

set -euo pipefail
cd /home/uace/dqn/atari
source /home/uace/dqn/.venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "SBATCH START $(date -Is)"
python train_nature.py \
  --env-id ALE/Breakout-v5 \
  --run-dir /home/uace/dqn/atari/runs/breakout_v3_smoke_seed20260621_rewardclip_dmdec100k_retry2 \
  --max-agent-steps-total 100000 \
  --seed 20260621 \
  --threads 8 \
  --diagnostics-every-agent-steps 10000 \
  --life-loss-terminal
echo "SBATCH END $(date -Is)"
