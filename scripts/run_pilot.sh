#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-run_$(date -u +%Y%m%d_%H%M%S)}"

python3 -m llm_affect_lab.pipeline.runner \
  --run-id "$RUN_ID" \
  --prompt-bank prompts/pilot.jsonl \
  --n-samples 5 \
  --max-tokens 1024

python3 -m llm_affect_lab.scoring.score_run \
  --run-id "$RUN_ID" \
  --models \
    meta-llama/llama-3.3-70b-instruct \
    google/gemma-3-27b-it \
    qwen/qwen-2.5-72b-instruct


echo "Pilot complete: $RUN_ID"
echo "Fingerprints: data/results/${RUN_ID}_fingerprints.json"
