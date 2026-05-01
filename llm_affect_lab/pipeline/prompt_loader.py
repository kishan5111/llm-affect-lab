"""Prompt-bank loading and validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from llm_affect_lab.storage.schema import PromptRecord


def load_prompt_bank(path: str | Path, *, shuffle: bool = False, seed: int = 42) -> list[PromptRecord]:
    prompt_path = Path(path)
    prompts = []
    with prompt_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                prompts.append(PromptRecord.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid prompt at {prompt_path}:{line_no}: {exc}") from exc

    ids = [prompt.id for prompt in prompts]
    duplicates = sorted({prompt_id for prompt_id in ids if ids.count(prompt_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate prompt ids in {prompt_path}: {duplicates}")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    return prompts

