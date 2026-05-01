#!/usr/bin/env python3
"""Summarize logprob probe coverage from raw JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def raw_path_for(model_slug: str, run_id: str) -> Path:
    return ROOT / "data" / "raw" / model_slug.replace("/", "__") / f"{run_id}.jsonl"


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{count}/{total} ({count / total:.0%})"


def summarize_model(model_slug: str, run_id: str) -> dict:
    rows = list(iter_jsonl(raw_path_for(model_slug, run_id)) or [])
    total = len(rows)
    top_logprobs_count = sum(
        1
        for row in rows
        if any((token.get("top_logprobs") or {}) for token in row.get("token_logprobs") or [])
    )
    self_report_count = sum(
        1
        for row in rows
        if ((row.get("self_report") or {}).get("weighted_score") is not None)
    )
    providers = sorted(
        {
            row.get("openrouter_provider_name")
            for row in rows
            if row.get("openrouter_provider_name")
        }
    )
    return {
        "model": model_slug,
        "rows": total,
        "logprobs": pct(sum(1 for row in rows if row.get("logprobs_available")), total),
        "top_logprobs": pct(top_logprobs_count, total),
        "self_report_weighted": pct(self_report_count, total),
        "reasoning": pct(sum(1 for row in rows if row.get("reasoning_text")), total),
        "length_finish": pct(sum(1 for row in rows if row.get("finish_reason") == "length"), total),
        "providers": ", ".join(providers) if providers else "unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    rows = [summarize_model(model, args.run_id) for model in args.models]
    headers = [
        "model",
        "rows",
        "logprobs",
        "top_logprobs",
        "self_report_weighted",
        "reasoning",
        "length_finish",
        "providers",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row[header]) for header in headers))


if __name__ == "__main__":
    main()
