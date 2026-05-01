#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-fullstudy160_first3_n5_4096_$(date -u +%Y%m%d_%H%M%S)}"

MODELS=(
  "openai/gpt-4o-mini"
  "deepseek/deepseek-v4-flash"
  "openai/gpt-oss-120b"
)

python3 -m llm_affect_lab.pipeline.runner \
  --run-id "$RUN_ID" \
  --prompt-bank prompts/full_study.jsonl \
  --n-samples 5 \
  --max-tokens 4096 \
  --concurrency 5 \
  --provider-preferences configs/provider_preferences.logprob_probe.json \
  --models "${MODELS[@]}"

python3 -m llm_affect_lab.scoring.score_run \
  --run-id "$RUN_ID" \
  --models "${MODELS[@]}"

python3 scripts/audit_pilot_results.py \
  --out-dir "data/audits/$RUN_ID" \
  --model-run "openai/gpt-4o-mini" "$RUN_ID" \
  --model-run "deepseek/deepseek-v4-flash" "$RUN_ID" \
  --model-run "openai/gpt-oss-120b" "$RUN_ID"

echo "Full-study first-3 run complete: $RUN_ID"
echo "Fingerprints: data/results/${RUN_ID}_fingerprints.json"
echo "Audit: data/audits/${RUN_ID}/pilot_audit.md"
