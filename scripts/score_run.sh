#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/score_run.sh RUN_ID [MODEL ...]" >&2
  exit 2
fi

RUN_ID="$1"
shift

if [[ $# -eq 0 ]]; then
  set -- \
    meta-llama/llama-3.3-70b-instruct \
    google/gemma-3-27b-it \
    qwen/qwen-2.5-72b-instruct
fi

python3 -m llm_affect_lab.scoring.score_run --run-id "$RUN_ID" --models "$@"
