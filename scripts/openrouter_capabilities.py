#!/usr/bin/env python3
"""Print OpenRouter support for logprobs/reasoning for selected models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline.openrouter_models import fetch_openrouter_models


DEFAULT_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-14b",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    models = fetch_openrouter_models()
    for slug in args.models:
        model = models.get(slug)
        if not model:
            print(f"{slug}: MISSING")
            continue
        params = set(model.get("supported_parameters") or [])
        print(f"{slug}:")
        print(f"  name: {model.get('name')}")
        print(f"  logprobs: {'yes' if {'logprobs', 'top_logprobs'} <= params else 'no'}")
        print(f"  reasoning: {'yes' if ('reasoning' in params or 'include_reasoning' in params) else 'no'}")
        print(f"  pricing: {model.get('pricing')}")
        print(f"  supported_parameters: {', '.join(model.get('supported_parameters') or [])}")


if __name__ == "__main__":
    main()
