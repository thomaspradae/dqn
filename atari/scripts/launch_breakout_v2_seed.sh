#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-20260609}"
RUN_ID="breakout_v2_seed${SEED}"
BASE="/home/uace/dqn/atari"
RUN_DIR="${BASE}/runs/${RUN_ID}"
LOG="${BASE}/logs/${RUN_ID}.out"
JOB_FILE="${BASE}/jobs/${RUN_ID}.sbatch"
REGISTRY="${BASE}/run_registry.csv"

mkdir -p "${BASE}/logs" "${BASE}/runs" "${BASE}/jobs"

if [ -d "${RUN_DIR}" ] && find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing to launch: run dir already exists and is non-empty: ${RUN_DIR}" >&2
    echo "choose a new seed/run id, or inspect/archive the existing run first" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}"

cat > "${JOB_FILE}" << EOF
#!/bin/bash
#SBATCH --job-name=breakout-v2
#SBATCH --partition=office
#SBATCH --nodelist=ofi1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=3-00:00:00
#SBATCH --output=${LOG}
#SBATCH --error=${BASE}/logs/${RUN_ID}.err

set -euo pipefail

cd ${BASE}
source /home/uace/dqn/.venv/bin/activate

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "SBATCH START \$(date -Is)"
echo "RUN=${RUN_ID}"
echo "SEED=${SEED}"
echo "HOST=\$(hostname)"
echo "PYTHON=\$(which python)"
free -h

python train_nature.py \\
  --env-id ALE/Breakout-v5 \\
  --run-dir "${RUN_DIR}" \\
  --episodes 100000 \\
  --max-raw-env-frames-total 50000000 \\
  --seed "${SEED}" \\
  --lr 2.5e-4 \\
  --gamma 0.99 \\
  --epsilon-start 1.0 \\
  --epsilon-end 0.1 \\
  --epsilon-decay 1000000 \\
  --batch-size 32 \\
  --target-update-freq 10000 \\
  --train-freq 4 \\
  --replay-size 1000000 \\
  --learning-starts 50000 \\
  --frame-skip 4 \\
  --noop-max 30 \\
  --checkpoint-every-episodes 100 \\
  --threads 8 \\
  --life-loss-terminal

free -h
echo "SBATCH END \$(date -Is)"
EOF

JOB_ID="$(sbatch --parsable "${JOB_FILE}")"

if [ ! -f "${REGISTRY}" ]; then
    cat > "${REGISTRY}" << 'CSV'
run_id,algo,game,env_id,owner_node,node,ssh_target,type,job_id,run_dir,log,seed,status,target_steps,step_unit,checkpoint_every,eval_every_steps,project_dir,venv_path,eval_script,checkpoint_pattern,metric_type,notes
CSV
fi

tmp_registry="${REGISTRY}.tmp"
grep -v "^${RUN_ID}," "${REGISTRY}" > "${tmp_registry}" || true
cat >> "${tmp_registry}" << CSV
${RUN_ID},scratch_dqn,Breakout,ALE/Breakout-v5,ofi1,ofi1,uace@100.107.98.78,scratch_dqn,${JOB_ID},${RUN_DIR},${LOG},${SEED},running,50000000,frames,100,5000000,${BASE},/home/uace/dqn/.venv,eval.py,q_net_ep{episode}.pt,scratch_rewards,v2 seed-controlled Breakout 50M
CSV
mv "${tmp_registry}" "${REGISTRY}"

echo "${RUN_ID}"
echo "${JOB_ID}"
echo "${RUN_DIR}"
echo "${LOG}"
