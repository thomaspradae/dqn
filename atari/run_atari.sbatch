#!/bin/bash
#SBATCH --job-name=dqn-pong
#SBATCH --output=/home/uace/dqn/atari/logs/dqn-pong-%j.out
#SBATCH --error=/home/uace/dqn/atari/logs/dqn-pong-%j.err
#SBATCH --partition=office
#SBATCH --nodelist=uaceofi1
#SBATCH --nodes=1
#SBATCH --time=INFINITE
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G

set -euo pipefail

cd /home/uace/dqn/atari

source /home/uace/dqn/.venv/bin/activate

mkdir -p logs runs

python train.py \
  --episodes 800 \
  --batch-size 32 \
  --replay-size 100000 \
  --outdir runs/pong_cpu_800
