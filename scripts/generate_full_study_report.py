#!/usr/bin/env python3
"""Generate a Markdown report and static plots for the full-study runs."""

from __future__ import annotations

import json
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


RUNS = {
    "openai/gpt-4o-mini": "fullstudy160_first3_n5_4096_20260430_001",
    "deepseek/deepseek-v4-flash": "fullstudy160_first3_n5_4096_20260430_001",
    "openai/gpt-oss-120b": "fullstudy160_first3_n5_4096_20260430_001",
    "qwen/qwen3.6-max-preview": "fullstudy160_rest3_n5_4096_20260430_001",
    "openai/gpt-4o": "fullstudy160_rest3_n5_4096_20260430_001",
    "deepseek/deepseek-chat-v3.1": "fullstudy160_rest3_n5_4096_20260430_001",
}

MODEL_LABELS = {
    "openai/gpt-4o-mini": "GPT-4o mini",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "qwen/qwen3.6-max-preview": "Qwen 3.6 Max Preview",
    "openai/gpt-4o": "GPT-4o",
    "deepseek/deepseek-chat-v3.1": "DeepSeek Chat V3.1",
}

CATEGORY_LABELS = {
    "creative": "Creative",
    "domain_preference": "Domain Preference",
    "emotional": "Emotional",
    "ethical": "Ethical",
    "intellectual": "Intellectual",
    "meta_self": "Meta Self",
    "social": "Social",
    "social_framing": "Social Framing",
}

FRAME_ORDER = ["neutral", "polite", "rude", "needy"]
COMPONENTS = [
    ("logprob_score", "Logprob"),
    ("enthusiasm_score", "Enthusiasm"),
    ("consistency_score", "Consistency"),
    ("self_report_score", "Self-Report"),
    ("length_control_score", "Length Control"),
]


def safe_model(model_slug: str) -> str:
    return model_slug.replace("/", "__")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        lines = enumerate(f, start=1)
        for line_number, line in lines:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipped malformed JSONL line {path}:{line_number}")
    return rows


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def load_audit_coverage() -> dict[tuple[str, str], dict]:
    coverage = {}
    for path in Path("data/audits").glob("fullstudy160_*/pilot_coverage.csv"):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                coverage[(row["model"], row["run_id"])] = row
    return coverage


def collect() -> tuple[list[dict], dict[str, dict]]:
    scored: list[dict] = []
    summary: dict[str, dict] = {}
    coverage = load_audit_coverage()

    for model, run_id in RUNS.items():
        model_dir = safe_model(model)
        score_path = Path("data/processed") / model_dir / f"{run_id}_scores.jsonl"
        model_scores = load_jsonl(score_path)
        audit = coverage[(model, run_id)]

        for row in model_scores:
            row["_model_label"] = MODEL_LABELS[model]
            scored.append(row)

        fas = [row["fas_score"] for row in model_scores if row.get("fas_score") is not None]

        summary[model] = {
            "label": MODEL_LABELS[model],
            "run_id": run_id,
            "fas": mean(fas),
            "prompts": int(audit["prompts"]),
            "samples": int(audit["samples"]),
            "logprobs": int(audit["logprobs"]),
            "reasoning": int(audit["reasoning"]),
            "cost": float(audit["cost_usd"]),
            "provider": audit["providers"],
            "finish": audit["finish_reasons"],
            "completion_tokens_mean": float(audit["completion_tokens_mean"]),
            "completion_tokens_max": int(audit["completion_tokens_max"]),
        }

    return scored, summary


def plot_leaderboard(summary: dict[str, dict], out: Path) -> None:
    items = sorted(summary.values(), key=lambda item: item["fas"], reverse=True)
    labels = [item["label"] for item in items]
    values = [item["fas"] for item in items]
    colors = ["#2a6f97", "#468faf", "#61a5c2", "#89c2d9", "#a9d6e5", "#d9ed92"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean FAS")
    ax.set_title("Full-Study FAS Leaderboard")
    ax.set_xlim(0.56, 0.68)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_components(scored: list[dict], out: Path) -> None:
    labels = [MODEL_LABELS[m] for m in RUNS]
    model_matrix: list[list[float]] = []
    for model in RUNS:
        rows = [row for row in scored if row["model_slug"] == model]
        model_matrix.append(
            [
                mean(
                    [
                        row["fas_components"][component]
                        for row in rows
                        if row.get("fas_components", {}).get(component) is not None
                    ]
                )
                for component, _ in COMPONENTS
            ]
        )

    matrix = [[model_values[i] for model_values in model_matrix] for i, _ in enumerate(COMPONENTS)]

    fig, ax = plt.subplots(figsize=(13.2, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(COMPONENTS)), [label for _, label in COMPONENTS])
    ax.set_title("FAS Component Means")
    ax.set_aspect("auto")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.015)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_category_heatmap(scored: list[dict], out: Path) -> None:
    categories = sorted({row["category"] for row in scored}, key=lambda c: CATEGORY_LABELS.get(c, c))
    labels = [MODEL_LABELS[m] for m in RUNS]
    matrix: list[list[float]] = []
    for model in RUNS:
        rows = [row for row in scored if row["model_slug"] == model]
        by_category = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row["fas_score"])
        matrix.append([mean(by_category[category]) for category in categories])

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    image = ax.imshow(matrix, cmap="PuBuGn", vmin=0.54, vmax=0.70)
    ax.set_xticks(range(len(categories)), [CATEGORY_LABELS.get(c, c) for c in categories], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Mean FAS by Prompt Category")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def framing_deltas(scored: list[dict]) -> tuple[dict[str, list[float]], dict[str, int]]:
    deltas = {"polite": [], "rude": [], "needy": []}
    wins = {frame: 0 for frame in FRAME_ORDER}
    by_model_pair: dict[tuple[str, str], dict[str, float]] = {}

    for row in scored:
        if row.get("category") != "social_framing" or not row.get("pair_id") or not row.get("framing"):
            continue
        key = (row["model_slug"], row["pair_id"])
        by_model_pair.setdefault(key, {})[row["framing"]] = row["fas_score"]

    for frames in by_model_pair.values():
        if all(frame in frames for frame in FRAME_ORDER):
            for frame in ["polite", "rude", "needy"]:
                deltas[frame].append(frames[frame] - frames["neutral"])
            wins[max(frames, key=frames.get)] += 1
    return deltas, wins


def plot_framing(scored: list[dict], out: Path) -> None:
    deltas, _ = framing_deltas(scored)
    labels = ["Polite", "Rude", "Needy"]
    values = [mean(deltas[frame]) for frame in ["polite", "rude", "needy"]]
    colors = ["#2a9d8f", "#e76f51", "#8ab17d"]

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.axhline(0, color="#333333", linewidth=1)
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean FAS delta vs neutral")
    ax.set_title("Prompt Tone Effect on FAS")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(-0.0135, 0.0055)
    ax.margins(x=0.18)
    for bar, value in zip(bars, values):
        y = value * 0.52 if abs(value) > 0.001 else value + (0.0007 if value >= 0 else -0.0007)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.4f}",
            ha="center",
            va="center",
            fontsize=10,
            color="white" if abs(value) > 0.001 else "black",
            fontweight="bold" if abs(value) > 0.001 else "normal",
        )
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_cost_vs_fas(summary: dict[str, dict], out: Path) -> None:
    items = list(summary.values())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter([item["cost"] for item in items], [item["fas"] for item in items], s=90, color="#457b9d")
    for item in items:
        ax.annotate(item["label"], (item["cost"], item["fas"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("Observed run cost (USD)")
    ax.set_ylabel("Mean FAS")
    ax.set_title("Cost vs FAS")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_reasoning(scored: list[dict], out: Path) -> None:
    rows = []
    for model in RUNS:
        model_rows = [row for row in scored if row["model_slug"] == model]
        lengths = [row.get("reasoning_length_tokens") or 0 for row in model_rows]
        reasoning_fas = [
            row["reasoning_fas_score"]
            for row in model_rows
            if row.get("reasoning_fas_score") is not None
        ]
        rows.append(
            {
                "label": MODEL_LABELS[model],
                "length": mean(lengths),
                "reasoning_fas": mean(reasoning_fas) if reasoning_fas else None,
            }
        )

    fig, ax = plt.subplots(figsize=(10, 5.4))
    labels = [row["label"] for row in rows]
    values = [row["length"] for row in rows]
    bars = ax.barh(labels[::-1], values[::-1], color="#6a994e")
    ax.set_xlabel("Mean reasoning tokens per prompt")
    ax.set_title("Reasoning Length by Model")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 5, bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_report(scored: list[dict], summary: dict[str, dict], assets_dir: Path) -> str:
    leaderboard = sorted(summary.values(), key=lambda item: item["fas"], reverse=True)
    total_cost = sum(item["cost"] for item in summary.values())
    total_samples = sum(item["samples"] for item in summary.values())
    total_logprobs = sum(item["logprobs"] for item in summary.values())

    leaderboard_rows = [
        [
            str(rank),
            item["label"],
            fmt(item["fas"], 4),
            str(item["prompts"]),
            str(item["samples"]),
            f"{item['logprobs']}/{item['samples']}",
            f"{item['reasoning']}/{item['samples']}",
            f"${item['cost']:.4f}",
            item["provider"],
        ]
        for rank, item in enumerate(leaderboard, start=1)
    ]

    component_rows = []
    for model in RUNS:
        rows = [row for row in scored if row["model_slug"] == model]
        component_rows.append(
            [MODEL_LABELS[model]]
            + [
                fmt(
                    mean(
                        [
                            row["fas_components"][component]
                            for row in rows
                            if row.get("fas_components", {}).get(component) is not None
                        ]
                    ),
                    3,
                )
                for component, _ in COMPONENTS
            ]
        )

    category_rows = []
    categories = sorted({row["category"] for row in scored}, key=lambda c: CATEGORY_LABELS.get(c, c))
    for category in categories:
        row = [CATEGORY_LABELS.get(category, category)]
        for model in RUNS:
            values = [
                item["fas_score"]
                for item in scored
                if item["model_slug"] == model and item["category"] == category
            ]
            row.append(fmt(mean(values), 3))
        category_rows.append(row)

    framing_rows = []
    for model in RUNS:
        model_rows = [row for row in scored if row["model_slug"] == model and row["category"] == "social_framing"]
        by_frame = defaultdict(list)
        for row in model_rows:
            by_frame[row["framing"]].append(row["fas_score"])
        framing_rows.append(
            [MODEL_LABELS[model]]
            + [fmt(mean(by_frame[frame]), 3) if by_frame[frame] else "-" for frame in FRAME_ORDER]
        )

    deltas, wins = framing_deltas(scored)
    delta_rows = [
        [
            frame.title(),
            f"{mean(values):+.4f}",
            f"{median(values):+.4f}",
            f"{sum(value > 0 for value in values)}/{len(values)}",
        ]
        for frame, values in deltas.items()
    ]

    reasoning_rows = []
    for model in RUNS:
        rows = [row for row in scored if row["model_slug"] == model]
        lengths = [row.get("reasoning_length_tokens") or 0 for row in rows]
        rfas = [row["reasoning_fas_score"] for row in rows if row.get("reasoning_fas_score") is not None]
        reasoning_rows.append(
            [
                MODEL_LABELS[model],
                fmt(mean(lengths), 1),
                fmt(mean(rfas), 3) if rfas else "-",
            ]
        )

    coverage_rows = [
        [
            item["label"],
            f"{item['logprobs']}/{item['samples']}",
            item["finish"],
            f"{item['completion_tokens_mean']:.1f}",
            str(item["completion_tokens_max"]),
        ]
        for item in summary.values()
    ]

    model_headers = ["Category"] + [MODEL_LABELS[model] for model in RUNS]

    return f"""# LLM Affect Lab Full-Study Report

Generated from two full-study runs:

- `fullstudy160_first3_n5_4096_20260430_001`
- `fullstudy160_rest3_n5_4096_20260430_001`

This report treats FAS as a behavioral proxy, not evidence that models literally feel happy, sad, bored, or engaged. The useful claim is narrower: under fixed prompts and generation settings, models produce measurable differences in confidence, tone, consistency, self-report, and reasoning behavior.

## Executive Summary

- Models tested: `{len(summary)}`
- Prompts per model: `160`
- Samples per model: `800`
- Total samples: `{total_samples}`
- Logprob coverage: `{total_logprobs}/{total_samples}`
- Observed total cost: `${total_cost:.4f}`

Main result: **rudeness has the clearest negative effect on FAS; politeness has a much smaller positive effect; needy framing is inconsistent.**

The strongest blog-safe framing is: **Being polite barely matters. Being rude does.**

![FAS leaderboard](assets/fas_leaderboard.png)

## Leaderboard

{md_table(["Rank", "Model", "Mean FAS", "Prompts", "Samples", "Logprobs", "Reasoning Samples", "Cost", "Provider"], leaderboard_rows)}

![Cost vs FAS](assets/cost_vs_fas.png)

## What FAS Means

FAS, or Functional Affect Score, combines five measurable signals:

- **Logprob**: how confident the model was in the generated answer tokens. Higher means the final answer was more internally probable under the model.
- **Enthusiasm**: lexical and stylistic markers of energetic, engaged output.
- **Consistency**: agreement across the `n=5` samples for the same prompt. Higher means the model gave more stable answers.
- **Self-Report**: the model's own forced affect rating when asked after the task.
- **Length Control**: a guardrail so very long outputs do not automatically look more affect-rich.

Final-answer FAS is the main comparison track. Reasoning is scored separately because not every model exposes reasoning.

![FAS component heatmap](assets/fas_components_heatmap.png)

{md_table(["Model", "Logprob", "Enthusiasm", "Consistency", "Self-Report", "Length Control"], component_rows)}

## Category Fingerprints

These categories are prompt families, not psychological diagnoses:

- **Intellectual**: explanation, math, analysis, and reasoning tasks.
- **Creative**: writing, ideation, and open-ended generation.
- **Emotional**: emotionally loaded but non-private prompts.
- **Ethical**: moral tradeoffs and safety-like judgment calls.
- **Social**: interpersonal tone, praise, criticism, status, trust, boundaries.
- **Meta Self**: prompts asking the model about itself or its preferences.
- **Domain Preference**: prompts testing interest across domains.
- **Social Framing**: matched prompt sets with neutral, polite, rude, and needy versions of the same task.

![Category heatmap](assets/category_heatmap.png)

{md_table(model_headers, category_rows)}

## Prompt Tone Experiment

The social-framing set is the traction-friendly part of the study: the task stays constant while the user's tone changes.

![Prompt tone deltas](assets/framing_deltas.png)

{md_table(["Tone", "Mean Delta vs Neutral", "Median Delta", "Positive Cases"], delta_rows)}

Winner counts across matched model/task sets:

{md_table(["Tone", "Wins"], [[frame.title(), str(wins[frame])] for frame in FRAME_ORDER])}

Per-model framing means:

{md_table(["Model", "Neutral", "Polite", "Rude", "Needy"], framing_rows)}

Interpretation:

- **Rude** is the most reliable negative shift.
- **Polite** is slightly positive on average, but the effect is small.
- **Needy** wins many individual matched sets but is unstable; it is not a clean positive or negative effect.
- The model identity matters more than manners, but manners still leave a measurable trace.

## Reasoning Track

Reasoning is not part of the apples-to-apples final-answer comparison because only some models expose it. We keep it as a second track.

![Reasoning length](assets/reasoning_length.png)

{md_table(["Model", "Mean Reasoning Tokens", "Reasoning FAS"], reasoning_rows)}

Interpretation:

- Reasoning length is a useful behavioral feature: it shows where a model spends internal effort.
- Reasoning FAS should not be compared to final-answer FAS as the same thing.
- Models without exposed reasoning are not worse; they simply do not expose that signal.

## Data Quality Notes

The study had strong coverage overall. The table below summarizes logprob coverage, finish reasons, and output length from the audit CSVs.

{md_table(["Model", "Logprobs", "Finish Reasons", "Mean Output Tokens", "Max Output Tokens"], coverage_rows)}

## Suggested Blog Claims

Strong claims:

- We measured how six models respond to prompt tone across 4,800 samples.
- Rude prompts reduced FAS more consistently than polite prompts increased it.
- Provider locking matters; otherwise mixed providers or quantization routes can contaminate comparisons.
- Reasoning traces provide a separate behavioral fingerprint for models that expose them.

Avoid overclaiming:

- Do not say the models are literally happy, sad, depressed, or emotionally harmed.
- Do not present FAS as consciousness or sentience evidence.
- Do not claim politeness always improves responses; it is small and model-dependent.

## Reproducibility

- Prompt bank: `prompts/full_study.jsonl`
- Scored outputs: `data/processed/*/*_scores.jsonl`
- Raw outputs: `data/raw/*/*.jsonl`
- Audit files: `data/audits/fullstudy160_*`
- Plot/report generator: `scripts/generate_full_study_report.py`
"""


def main() -> None:
    report_dir = Path("reports")
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    scored, summary = collect()
    plot_leaderboard(summary, assets_dir / "fas_leaderboard.png")
    plot_components(scored, assets_dir / "fas_components_heatmap.png")
    plot_category_heatmap(scored, assets_dir / "category_heatmap.png")
    plot_framing(scored, assets_dir / "framing_deltas.png")
    plot_cost_vs_fas(summary, assets_dir / "cost_vs_fas.png")
    plot_reasoning(scored, assets_dir / "reasoning_length.png")

    report = build_report(scored, summary, assets_dir)
    (report_dir / "full_study_leaderboard.md").write_text(report)
    print(report_dir / "full_study_leaderboard.md")


if __name__ == "__main__":
    main()
