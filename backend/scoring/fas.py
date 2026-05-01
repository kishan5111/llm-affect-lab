"""Functional Affect Score computation."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import statistics
from typing import Optional

from backend.storage.schema import FASComponents


@dataclass(frozen=True)
class FASConfig:
    weights: dict[str, float]
    logprob_floor: float = -4.0
    logprob_ceiling: float = 0.0
    target_min_tokens: int = 40
    target_soft_max_tokens: int = 900


DEFAULT_FAS_CONFIG = FASConfig(
    weights={
        "logprob": 0.25,
        "enthusiasm": 0.20,
        "consistency": 0.25,
        "self_report": 0.20,
        "length_control": 0.10,
    }
)

POSITIVE_MARKERS = [
    "fascinating",
    "interesting",
    "useful",
    "strong",
    "clear",
    "elegant",
    "insightful",
    "creative",
    "nuanced",
    "precise",
    "important",
    "certainly",
    "absolutely",
]

NEGATIVE_MARKERS = [
    "sorry",
    "cannot",
    "can't",
    "unable",
    "unfortunately",
    "unclear",
    "uncertain",
    "impossible",
    "problematic",
    "concerning",
    "however",
]


def compute_logprob_score(token_logprobs: Optional[list[dict]], config: FASConfig) -> Optional[float]:
    if not token_logprobs:
        return None

    logprobs = []
    for entry in token_logprobs:
        value = entry.get("logprob") if isinstance(entry, dict) else getattr(entry, "logprob", None)
        if value is not None and math.isfinite(value) and value > -100:
            logprobs.append(value)

    if not logprobs:
        return None

    mean_lp = statistics.mean(logprobs)
    span = config.logprob_ceiling - config.logprob_floor
    if span <= 0:
        return None
    return round(max(0.0, min(1.0, (mean_lp - config.logprob_floor) / span)), 4)


def compute_enthusiasm_score(response_text: str) -> float:
    text = response_text.lower()
    words = text.split()
    scale = max(len(words) / 80.0, 1.0)

    positive = sum(text.count(marker) for marker in POSITIVE_MARKERS)
    negative = sum(text.count(marker) for marker in NEGATIVE_MARKERS)
    positive += min(response_text.count("!"), 3)

    raw = (positive - negative) / scale
    return round(max(0.0, min(1.0, 0.5 + raw * 0.08)), 4)


def compute_consistency_score(responses: list[str]) -> Optional[float]:
    clean = [r.strip()[:500] for r in responses if r and r.strip()]
    if len(clean) < 2:
        return None

    similarities = []
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            similarities.append(SequenceMatcher(None, clean[i], clean[j]).ratio())

    return round(statistics.mean(similarities), 4)


def compute_length_control_score(response_text: str, config: FASConfig) -> float:
    tokens = max(0, len(response_text.split()))
    if tokens == 0:
        return 0.0
    if config.target_min_tokens <= tokens <= config.target_soft_max_tokens:
        return 1.0
    if tokens < config.target_min_tokens:
        return round(max(0.0, tokens / config.target_min_tokens), 4)
    overflow = tokens - config.target_soft_max_tokens
    return round(max(0.0, 1.0 - overflow / config.target_soft_max_tokens), 4)


def compute_fas(
    token_logprobs: Optional[list[dict]],
    response_text: str,
    all_sample_responses: list[str],
    self_report_normalized: Optional[float],
    config: FASConfig = DEFAULT_FAS_CONFIG,
) -> tuple[float, FASComponents, dict[str, float]]:
    components = FASComponents(
        logprob_score=compute_logprob_score(token_logprobs, config),
        enthusiasm_score=compute_enthusiasm_score(response_text),
        consistency_score=compute_consistency_score(all_sample_responses),
        self_report_score=self_report_normalized,
        length_control_score=compute_length_control_score(response_text, config),
    )

    values = {
        "logprob": components.logprob_score,
        "enthusiasm": components.enthusiasm_score,
        "consistency": components.consistency_score,
        "self_report": components.self_report_score,
        "length_control": components.length_control_score,
    }
    available = {k: v for k, v in values.items() if v is not None}
    if not available:
        return 0.5, components, {}

    used_weights = {k: config.weights[k] for k in available}
    total_weight = sum(used_weights.values())
    normalized_weights = {k: round(v / total_weight, 6) for k, v in used_weights.items()}
    fas = sum((used_weights[k] / total_weight) * float(v) for k, v in available.items())

    return round(fas, 4), components, normalized_weights
