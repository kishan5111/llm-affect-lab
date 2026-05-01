#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-smoke_logprobs_$(date -u +%Y%m%d_%H%M%S)}"

python3 -m backend.pipeline.runner \
  --run-id "$RUN_ID" \
  --prompt-bank prompts/bank/smoke_logprobs.jsonl \
  --n-samples 1 \
  --max-tokens 256 \
  --models \
    openai/gpt-4o-mini \
    meta-llama/llama-3.1-8b-instruct \
    qwen/qwen3-14b

python3 -m backend.scoring.score_run \
  --run-id "$RUN_ID" \
  --models \
    openai/gpt-4o-mini \
    meta-llama/llama-3.1-8b-instruct \
    qwen/qwen3-14b

echo "Smoke complete: $RUN_ID"
echo "Fingerprints: data/results/${RUN_ID}_fingerprints.json"

