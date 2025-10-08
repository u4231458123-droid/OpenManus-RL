#!/usr/bin/env bash
set -euo pipefail

# Ensure we start at repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load API keys if .env exists
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="experiments/smoke_gaia_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

python scripts/rollout/openmanus_rollout_debugger.py \
  --env gaia \
  --model "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --debugger_model gpt-4.1 \
  --together rollout \
  --total_envs 1 \
  --test_times 1 \
  --max_steps 6 \
  --history_length 6 \
  --temperature 0.0 \
  --concurrency 1 \
  --llm_concurrency 1 \
  --max_try 3 \
  --enable_debugger \
  --experiment_dir "${RUN_DIR}" \
  --dump_path "${RUN_DIR}/summaries/trajectory.jsonl" \
  --save_per_task_trajectories \
  --unique_envs \
  --debug
