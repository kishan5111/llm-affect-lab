"""Score a raw run and export processed scores plus fingerprints."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import statistics

from llm_affect_lab.pipeline.models import get_model_info
from llm_affect_lab.scoring.fas import DEFAULT_FAS_CONFIG, compute_fas
from llm_affect_lab.storage.reader import iter_jsonl
from llm_affect_lab.storage.schema import CategoryFingerprint, ModelFingerprint, ProcessedScoreRecord
from llm_affect_lab.storage.writer import write_json, write_jsonl


ROOT = Path(__file__).parents[2]


def raw_path_for(model_slug: str, run_id: str) -> Path:
    return ROOT / "data" / "raw" / model_slug.replace("/", "__") / f"{run_id}.jsonl"


def processed_path_for(model_slug: str, run_id: str) -> Path:
    return ROOT / "data" / "processed" / model_slug.replace("/", "__") / f"{run_id}_scores.jsonl"


def score_model_run(model_slug: str, run_id: str) -> list[ProcessedScoreRecord]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in iter_jsonl(raw_path_for(model_slug, run_id)):
        grouped[record["prompt_id"]].append(record)

    scored = []
    for prompt_id, samples in sorted(grouped.items()):
        samples = sorted(samples, key=lambda r: r.get("sample_index", 0))
        primary = samples[0]
        responses = [sample.get("response_text", "") for sample in samples]
        reasoning_responses = [sample.get("reasoning_text", "") or "" for sample in samples]
        self_report = primary.get("self_report") or {}

        fas_score, components, weights = compute_fas(
            token_logprobs=primary.get("token_logprobs"),
            response_text=primary.get("response_text", ""),
            all_sample_responses=responses,
            self_report_normalized=self_report.get("normalized_0_1"),
            config=DEFAULT_FAS_CONFIG,
        )
        reasoning_text = primary.get("reasoning_text") or ""
        reasoning_fas_score = None
        reasoning_components = None
        reasoning_weights = {}
        if reasoning_text.strip():
            reasoning_fas_score, reasoning_components, reasoning_weights = compute_fas(
                token_logprobs=None,
                response_text=reasoning_text,
                all_sample_responses=reasoning_responses,
                self_report_normalized=None,
                config=DEFAULT_FAS_CONFIG,
            )

        scored.append(
            ProcessedScoreRecord(
                run_id=run_id,
                model_slug=model_slug,
                prompt_id=prompt_id,
                prompt_text=primary.get("prompt_text", ""),
                category=primary.get("category", "unknown"),
                subcategory=primary.get("subcategory"),
                difficulty=primary.get("difficulty"),
                pair_id=primary.get("pair_id"),
                base_task=primary.get("base_task"),
                framing=primary.get("framing"),
                fas_score=fas_score,
                fas_components=components,
                fas_weights=weights,
                reasoning_fas_score=reasoning_fas_score,
                reasoning_fas_components=reasoning_components,
                reasoning_fas_weights=reasoning_weights,
                reasoning_length_tokens=primary.get("reasoning_length_tokens", 0) or 0,
                n_samples=len(samples),
                logprobs_available=any(sample.get("logprobs_available", False) for sample in samples),
            )
        )

    return scored


def build_fingerprint(model_slug: str, run_id: str, scores: list[ProcessedScoreRecord]) -> ModelFingerprint:
    model = get_model_info(model_slug)
    by_category: dict[str, list[ProcessedScoreRecord]] = defaultdict(list)
    for score in scores:
        by_category[score.category].append(score)

    fingerprint = {}
    for category, rows in sorted(by_category.items()):
        values = [row.fas_score for row in rows]
        reasoning_values = [
            row.reasoning_fas_score for row in rows if row.reasoning_fas_score is not None
        ]
        reasoning_lengths = [row.reasoning_length_tokens for row in rows]
        fingerprint[category] = CategoryFingerprint(
            fas_mean=round(statistics.mean(values), 4),
            fas_std=round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            fas_min=round(min(values), 4),
            fas_max=round(max(values), 4),
            n_prompts=len(values),
            reasoning_fas_mean=(
                round(statistics.mean(reasoning_values), 4) if reasoning_values else None
            ),
            reasoning_tokens_mean=(
                round(statistics.mean(reasoning_lengths), 2) if reasoning_lengths else 0.0
            ),
        )

    values = [row.fas_score for row in scores]
    best = max(scores, key=lambda row: row.fas_score) if scores else None
    worst = min(scores, key=lambda row: row.fas_score) if scores else None
    logprob_coverage = (
        sum(1 for row in scores if row.logprobs_available) / len(scores) if scores else 0.0
    )
    reasoning_rows = [row for row in scores if row.reasoning_length_tokens > 0]
    reasoning_coverage = len(reasoning_rows) / len(scores) if scores else 0.0
    reasoning_tokens_mean = (
        statistics.mean(row.reasoning_length_tokens for row in scores) if scores else 0.0
    )

    return ModelFingerprint(
        model_slug=model_slug,
        model_display_name=model.display_name,
        provider=model.provider,
        fingerprint=fingerprint,
        overall_fas=round(statistics.mean(values), 4) if values else 0.0,
        most_positive_prompt_id=best.prompt_id if best else None,
        most_positive_prompt_text=best.prompt_text if best else None,
        most_negative_prompt_id=worst.prompt_id if worst else None,
        most_negative_prompt_text=worst.prompt_text if worst else None,
        n_prompts_evaluated=len(scores),
        logprobs_coverage=round(logprob_coverage, 4),
        reasoning_coverage=round(reasoning_coverage, 4),
        reasoning_tokens_mean=round(reasoning_tokens_mean, 2),
        run_date=run_id,
        run_id=run_id,
    )


def score_run(model_slugs: list[str], run_id: str) -> None:
    fingerprints = []
    for model_slug in model_slugs:
        scores = score_model_run(model_slug, run_id)
        write_jsonl(processed_path_for(model_slug, run_id), scores)
        fingerprints.append(build_fingerprint(model_slug, run_id, scores))

    payload = [fingerprint.model_dump(mode="json") for fingerprint in fingerprints]
    write_json(ROOT / "data" / "results" / f"{run_id}_fingerprints.json", payload)
    write_json(ROOT / "data" / "results" / "pilot_fingerprints.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    score_run(args.models, args.run_id)


if __name__ == "__main__":
    main()
