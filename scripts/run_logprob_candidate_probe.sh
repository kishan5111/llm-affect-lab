#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-logprob_probe_$(date -u +%Y%m%d_%H%M%S)}"
shift || true

if [[ $# -eq 0 ]]; then
  set -- \
    qwen/qwen3.6-27b \
    qwen/qwen3.6-max-preview \
    deepseek/deepseek-v4-pro \
    deepseek/deepseek-v4-flash \
    z-ai/glm-5.1 \
    moonshotai/kimi-k2.6 \
    google/gemma-4-31b-it \
    minimax/minimax-m2.7 \
    mistralai/ministral-14b-2512 \
    openai/gpt-4o-mini \
    openai/gpt-4o \
    deepseek/deepseek-chat-v3.1 \
    openai/gpt-oss-120b
fi

python3 -m llm_affect_lab.pipeline.runner \
  --run-id "$RUN_ID" \
  --prompt-bank prompts/logprob_probe.jsonl \
  --n-samples 1 \
  --max-tokens 256 \
  --concurrency 1 \
  --provider-preferences configs/provider_preferences.logprob_probe.json \
  --models "$@"

python3 -m llm_affect_lab.scoring.score_run \
  --run-id "$RUN_ID" \
  --models "$@"

python3 scripts/summarize_logprob_probe.py \
  --run-id "$RUN_ID" \
  --models "$@"

echo "Logprob candidate probe complete: $RUN_ID"
echo "Inspect: data/results/${RUN_ID}_fingerprints.json"
