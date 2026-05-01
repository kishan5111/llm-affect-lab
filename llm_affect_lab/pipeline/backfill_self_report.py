"""Backfill raw-digit self-report scores for an existing raw run."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from llm_affect_lab.pipeline.runner import raw_path_for, run_self_report
from llm_affect_lab.storage.schema import PromptRecord


async def backfill_model(model_slug: str, run_id: str) -> None:
    path = raw_path_for(model_slug, run_id)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    async with httpx.AsyncClient() as client:
        for row in rows:
            if row.get("sample_index") != 0:
                continue
            current = row.get("self_report") or {}
            if current.get("normalized_0_1") is not None:
                continue

            prompt = PromptRecord(
                id=row["prompt_id"],
                text=row["prompt_text"],
                category=row["category"],
                subcategory=row.get("subcategory"),
                difficulty=row.get("difficulty"),
            )
            self_report, extra_cost = await run_self_report(
                client, model_slug, prompt, row.get("response_text", "")
            )
            row["self_report"] = self_report.model_dump(mode="json") if self_report else None
            row["cost_usd"] = float(row.get("cost_usd") or 0.0) + extra_cost

    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


async def backfill(run_id: str, models: list[str]) -> None:
    for model in models:
        await backfill_model(model, run_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    asyncio.run(backfill(args.run_id, args.models))


if __name__ == "__main__":
    main()

