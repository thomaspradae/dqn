#!/bin/bash
#SBATCH --job-name=dqn-cartpole
#SBATCH --output=/home/uace/jobs/dqn-cartpole/logs/slurm-%j.out
#SBATCH --partition=debug
#SBATCH --nodelist=ofi1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

cd /home/uace/jobs/dqn-cartpole

source .venv/bin/activate
python train.py
