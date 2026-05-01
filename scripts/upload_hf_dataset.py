#!/usr/bin/env python3
"""Build and upload the dataset package to Hugging Face."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, create_repo, upload_folder


DEFAULT_REPO_ID = "kishan51/llm-affect-lab"
EXPORT_DIR = Path("data/hf_export/llm-affect-lab")

FULL_STUDY_RUNS = {
    "fullstudy160_first3_n5_4096_20260430_001",
    "fullstudy160_rest3_n5_4096_20260430_001",
}


def get_token() -> str:
    env = dotenv_values(".env")
    token = (
        env.get("HF_TOKEN")
        or env.get("HUGGINGFACE_TOKEN")
        or env.get("HUGGING_FACE_HUB_TOKEN")
        or env.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError("No Hugging Face token found in .env or environment")
    return token


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sanitize_jsonl(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    skipped = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            valid += 1
    return valid, skipped


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def copy_tree_files(src_root: Path, dst_root: Path, patterns: list[str]) -> None:
    for pattern in patterns:
        for src in src_root.glob(pattern):
            if src.is_file():
                copy_file(src, dst_root / src.relative_to(src_root))


def build_dataset_card(repo_id: str) -> str:
    return f"""---
license: cc-by-4.0
pretty_name: LLM Affect Lab
task_categories:
- text-generation
language:
- en
tags:
- llm
- prompt-engineering
- logprobs
- evaluation
- affective-computing
- openrouter
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: full_study_samples.csv
---

# LLM Affect Lab

This dataset contains the API-level results for **LLM Affect Lab**, a study of functional affect signatures in language model behavior.

Functional Affect Score (FAS) is a 0-1 behavioral proxy. It combines generated-token confidence, enthusiastic language, consistency across repeated samples, forced self-report, and length control. The goal is not to claim that models feel emotions; the goal is to measure whether different prompt styles leave systematic behavioral traces in model outputs.

This run tests prompts across intellectual, creative, social-framing, existential, and practical categories, including polite, rude, and needy variants. The main browseable table is `full_study_samples.csv`.

Repository: https://github.com/kishan51/llm-affect-lab

Dataset path: https://huggingface.co/datasets/{repo_id}

## Contents

- `raw/`: raw model response JSONL files for the two full-study runs.
- `processed/`: scored per-prompt JSONL files with Functional Affect Score components.
- `full_study_samples.csv`: sample-level table for browsing in the Hugging Face dataset viewer.
- `results/`: aggregate fingerprint JSON files.
- `prompts/`: prompt banks used for smoke, pilot, probe, and full-study runs.

## Study Shape

- Models: 6
- Prompts per model: 160
- Samples per prompt: 5
- Total responses: 4,800
- Temperature: 1.0
- Top-p: 1.0
- Max output tokens: 4,096
- Logprob coverage: 4,797 / 4,800

## Main Finding

The safest summary is:

> Rude prompts reduced the affect proxy more consistently than polite prompts increased it.

This is not evidence that LLMs literally feel emotion. FAS is a behavioral proxy based on output confidence, style, consistency, self-report, and length control.

## Citation

```bibtex
@misc{{vavdara2026llmaffectlab,
  title        = {{LLM Affect Lab: Measuring Functional Affect Signatures in Language Model Behavior}},
  author       = {{Kishan Vavdara}},
  year         = {{2026}},
  howpublished = {{Hugging Face dataset}},
  url          = {{https://huggingface.co/datasets/{repo_id}}},
  note         = {{Code: https://github.com/kishan51/llm-affect-lab}}
}}
```

## Links

- Dataset: https://huggingface.co/datasets/{repo_id}
- Code: https://github.com/kishan51/llm-affect-lab
- Copyright: 2026 Kishan Vavdara
"""


def load_score_index() -> dict[tuple[str, str, str], dict]:
    scores = {}
    for src in Path("data/processed").glob("*/*_scores.jsonl"):
        if not any(run_id in src.name for run_id in FULL_STUDY_RUNS):
            continue
        for row in iter_jsonl(src):
            key = (row["run_id"], row["model_slug"], row["prompt_id"])
            scores[key] = row
    return scores


def write_sample_table(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    scores = load_score_index()
    fields = [
        "run_id",
        "model_slug",
        "provider",
        "prompt_id",
        "category",
        "subcategory",
        "difficulty",
        "framing",
        "sample_index",
        "prompt_text",
        "generated_text",
        "mean_logprob",
        "logprobs_available",
        "fas_score",
        "logprob_score",
        "enthusiasm_score",
        "consistency_score",
        "self_report_score",
        "length_control_score",
        "reasoning_fas_score",
        "reasoning_length_tokens",
        "response_length_tokens",
        "self_report_raw_digit",
        "finish_reason",
    ]

    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for src in sorted(Path("data/raw").glob("*/*.jsonl")):
            if src.stem not in FULL_STUDY_RUNS:
                continue
            for raw in iter_jsonl(src):
                score = scores.get((raw["run_id"], raw["model_slug"], raw["prompt_id"]), {})
                components = score.get("fas_components") or {}
                self_report = raw.get("self_report") or {}
                writer.writerow(
                    {
                        "run_id": raw.get("run_id"),
                        "model_slug": raw.get("model_slug"),
                        "provider": raw.get("openrouter_provider_name"),
                        "prompt_id": raw.get("prompt_id"),
                        "category": raw.get("category"),
                        "subcategory": raw.get("subcategory"),
                        "difficulty": raw.get("difficulty"),
                        "framing": raw.get("framing"),
                        "sample_index": raw.get("sample_index"),
                        "prompt_text": raw.get("prompt_text"),
                        "generated_text": raw.get("response_text"),
                        "mean_logprob": raw.get("mean_logprob"),
                        "logprobs_available": raw.get("logprobs_available"),
                        "fas_score": score.get("fas_score"),
                        "logprob_score": components.get("logprob_score"),
                        "enthusiasm_score": components.get("enthusiasm_score"),
                        "consistency_score": components.get("consistency_score"),
                        "self_report_score": components.get("self_report_score"),
                        "length_control_score": components.get("length_control_score"),
                        "reasoning_fas_score": score.get("reasoning_fas_score"),
                        "reasoning_length_tokens": raw.get("reasoning_length_tokens"),
                        "response_length_tokens": raw.get("response_length_tokens"),
                        "self_report_raw_digit": self_report.get("raw_digit"),
                        "finish_reason": raw.get("finish_reason"),
                    }
                )


def build_export(repo_id: str) -> Path:
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "repo_id": repo_id,
        "runs": sorted(FULL_STUDY_RUNS),
        "raw_files": [],
        "processed_files": [],
        "sanitization": [],
    }

    for src in Path("data/raw").glob("*/*.jsonl"):
        if src.stem not in FULL_STUDY_RUNS:
            continue
        dst = EXPORT_DIR / "raw" / src.parent.name / src.name
        valid, skipped = sanitize_jsonl(src, dst)
        manifest["raw_files"].append(str(dst.relative_to(EXPORT_DIR)))
        manifest["sanitization"].append(
            {"source": str(src), "target": str(dst.relative_to(EXPORT_DIR)), "valid_rows": valid, "skipped_rows": skipped}
        )

    for src in Path("data/processed").glob("*/*_scores.jsonl"):
        if not any(run_id in src.name for run_id in FULL_STUDY_RUNS):
            continue
        dst = EXPORT_DIR / "processed" / src.parent.name / src.name
        valid, skipped = sanitize_jsonl(src, dst)
        manifest["processed_files"].append(str(dst.relative_to(EXPORT_DIR)))
        manifest["sanitization"].append(
            {"source": str(src), "target": str(dst.relative_to(EXPORT_DIR)), "valid_rows": valid, "skipped_rows": skipped}
        )

    copy_tree_files(Path("data/results"), EXPORT_DIR / "results", ["fullstudy160_*"])
    copy_tree_files(Path("prompts/bank"), EXPORT_DIR / "prompts/bank", ["*.jsonl"])
    write_sample_table(EXPORT_DIR / "full_study_samples.csv")

    (EXPORT_DIR / "README.md").write_text(build_dataset_card(repo_id), encoding="utf-8")
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return EXPORT_DIR


def upload(repo_id: str, private: bool) -> None:
    token = get_token()
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", private=private, token=token, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(EXPORT_DIR),
        path_in_repo=".",
        token=token,
        commit_message="Upload LLM Affect Lab dataset",
    )
    info = api.dataset_info(repo_id, token=token)
    print(f"Uploaded dataset: https://huggingface.co/datasets/{info.id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    export_dir = build_export(args.repo_id)
    print(f"Built dataset export: {export_dir}")
    if not args.build_only:
        upload(args.repo_id, args.private)


if __name__ == "__main__":
    main()
