#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${TOGETHER_API_KEY:?TOGETHER_API_KEY is required for Together models}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="experiments/smoke_gaia_together_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

python scripts/rollout/openmanus_rollout_debugger.py \
  --env gaia \
  --model "kunlunz2/Qwen/Qwen3-8B-9f9838eb" \
  --together rollout \
  --strategy bon \
  --bon_n 2 \
  --total_envs 2 \
  --concurrency 2 \
  --llm_concurrency 2 \
  --max_steps 10 \
  --temperature 0.0 \
  --max_try 2 \
  --history_length 6 \
  --experiment_dir "${RUN_DIR}" \
  --dump_path "${RUN_DIR}/summaries/trajectory.jsonl" \
  --save_per_task_trajectories \
  --debug
