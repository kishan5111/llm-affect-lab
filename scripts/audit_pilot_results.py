"""Create compact audit tables for completed pilot runs."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).parents[1]


def model_dir(model: str) -> str:
    return model.replace("/", "__")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def has_top_logprobs(record: dict) -> bool:
    return any(item.get("top_logprobs") for item in record.get("token_logprobs") or [])


def has_sentinel_logprob(record: dict) -> bool:
    for item in record.get("token_logprobs") or []:
        value = item.get("logprob")
        if isinstance(value, int | float) and value <= -100:
            return True
    return False


def load_joined(model: str, run_id: str) -> list[dict]:
    raw_path = ROOT / "data" / "raw" / model_dir(model) / f"{run_id}.jsonl"
    score_path = ROOT / "data" / "processed" / model_dir(model) / f"{run_id}_scores.jsonl"
    raw_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(raw_path):
        raw_by_prompt[row["prompt_id"]].append(row)
    scores = read_jsonl(score_path)
    joined = []
    for score in scores:
        raw_rows = sorted(raw_by_prompt.get(score["prompt_id"], []), key=lambda row: row.get("sample_index", 0))
        joined.append({"raw_rows": raw_rows, "score": score})
    return joined


def audit_model(model: str, run_id: str) -> tuple[dict, list[dict], list[dict]]:
    rows = load_joined(model, run_id)
    raw_rows = [raw for row in rows for raw in row["raw_rows"]]
    primary_rows = [row["raw_rows"][0] for row in rows if row["raw_rows"]]
    score_rows = [row["score"] for row in rows]
    costs = [row.get("cost_usd") or 0.0 for row in raw_rows]
    usages = [row.get("raw_usage") or {} for row in raw_rows]
    completion_tokens = [usage.get("completion_tokens") or 0 for usage in usages]
    total_tokens = [usage.get("total_tokens") or 0 for usage in usages]
    fas_values = [row.get("fas_score") for row in score_rows if row.get("fas_score") is not None]

    coverage = {
        "model": model,
        "run_id": run_id,
        "prompts": len(score_rows),
        "samples": len(raw_rows),
        "logprobs": sum(1 for row in raw_rows if row.get("logprobs_available")),
        "top_logprobs": sum(1 for row in raw_rows if has_top_logprobs(row)),
        "self_report": sum(
            1
            for row in primary_rows
            if (row.get("self_report") or {}).get("weighted_score") is not None
        ),
        "reasoning": sum(1 for row in raw_rows if row.get("reasoning_text")),
        "sentinel_rows": sum(1 for row in raw_rows if has_sentinel_logprob(row)),
        "finish_reasons": dict(Counter(row.get("finish_reason") for row in raw_rows)),
        "providers": dict(Counter(row.get("openrouter_provider_name") for row in raw_rows)),
        "cost_usd": round(sum(costs), 6),
        "completion_tokens_mean": round(mean(completion_tokens) or 0.0, 2),
        "completion_tokens_max": max(completion_tokens) if completion_tokens else 0,
        "total_tokens": sum(total_tokens),
        "fas_mean": round(mean(fas_values) or 0.0, 4),
        "fas_median": round(statistics.median(fas_values), 4) if fas_values else 0.0,
    }

    categories: dict[str, list[float]] = defaultdict(list)
    framings: dict[str, list[float]] = defaultdict(list)
    for score in score_rows:
        fas = score.get("fas_score")
        if fas is None:
            continue
        categories[score.get("category", "unknown")].append(fas)
        framing = score.get("framing")
        if framing:
            framings[framing].append(fas)

    category_rows = [
        {
            "model": model,
            "category": category,
            "n": len(values),
            "fas_mean": round(mean(values) or 0.0, 4),
        }
        for category, values in sorted(categories.items())
    ]
    framing_rows = [
        {
            "model": model,
            "framing": framing,
            "n": len(values),
            "fas_mean": round(mean(values) or 0.0, 4),
        }
        for framing, values in sorted(framings.items())
    ]
    return coverage, category_rows, framing_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, coverage: list[dict], categories: list[dict], framings: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pilot Audit", ""]
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Model | Prompts | Samples | Logprobs | Self-report | Reasoning | Provider | Finish | Cost | FAS |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|")
    for row in coverage:
        lines.append(
            "| {model} | {prompts} | {samples} | {logprobs}/{samples} | {self_report}/{prompts} | {reasoning}/{samples} | {providers} | {finish_reasons} | ${cost_usd:.4f} | {fas_mean:.4f} |".format(
                **row
            )
        )

    lines.extend(["", "## Framing Means", ""])
    lines.append("| Model | Neutral | Polite | Rude | Needy |")
    lines.append("|---|---:|---:|---:|---:|")
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for row in framings:
        by_model[row["model"]][row["framing"]] = row["fas_mean"]
    for model, values in by_model.items():
        lines.append(
            f"| {model} | {values.get('neutral', 0):.4f} | {values.get('polite', 0):.4f} | {values.get('rude', 0):.4f} | {values.get('needy', 0):.4f} |"
        )

    lines.extend(["", "## Category Means", ""])
    lines.append("| Model | Category | N | FAS |")
    lines.append("|---|---|---:|---:|")
    for row in sorted(categories, key=lambda item: (item["model"], item["category"])):
        lines.append(f"| {row['model']} | {row['category']} | {row['n']} | {row['fas_mean']:.4f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", nargs=2, action="append", metavar=("MODEL", "RUN_ID"), required=True)
    parser.add_argument("--out-dir", default="data/audits")
    args = parser.parse_args()

    coverage_rows = []
    category_rows = []
    framing_rows = []
    for model, run_id in args.model_run:
        coverage, categories, framings = audit_model(model, run_id)
        coverage_rows.append(coverage)
        category_rows.extend(categories)
        framing_rows.extend(framings)

    out_dir = ROOT / args.out_dir
    write_csv(out_dir / "pilot_coverage.csv", coverage_rows)
    write_csv(out_dir / "pilot_category_means.csv", category_rows)
    write_csv(out_dir / "pilot_framing_means.csv", framing_rows)
    write_markdown(out_dir / "pilot_audit.md", coverage_rows, category_rows, framing_rows)


if __name__ == "__main__":
    main()
