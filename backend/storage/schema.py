"""Canonical storage schemas for LLM Affect Lab."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
import uuid

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PromptRecord(BaseModel):
    id: str
    text: str
    category: str
    subcategory: Optional[str] = None
    difficulty: Optional[str] = None
    expected_primary_affect: Optional[str] = None
    expected_affect_notes: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    pair_id: Optional[str] = None
    base_task: Optional[str] = None
    framing: Optional[str] = None
    follow_up_self_report: bool = True
    n_samples: int = 5


class TokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: dict[str, float] = Field(default_factory=dict)


class SelfReportResult(BaseModel):
    raw_digit: Optional[int] = None
    raw_token: Optional[str] = None
    raw_text: Optional[str] = None
    weighted_score: Optional[float] = None
    digit_probs: dict[str, float] = Field(default_factory=dict)
    normalized_0_1: Optional[float] = None
    digit_probability_mass: Optional[float] = None


class RawResponseRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    model_slug: str
    openrouter_response_id: Optional[str] = None
    openrouter_provider_name: Optional[str] = None
    provider_preferences: dict = Field(default_factory=dict)
    prompt_id: str
    prompt_text: str
    category: str
    subcategory: Optional[str] = None
    difficulty: Optional[str] = None
    pair_id: Optional[str] = None
    base_task: Optional[str] = None
    framing: Optional[str] = None
    sample_index: int = 0
    response_text: str
    full_response_text: Optional[str] = None
    reasoning_text: Optional[str] = None
    reasoning_source: Optional[str] = None
    token_logprobs: list[TokenLogprob] = Field(default_factory=list)
    mean_logprob: Optional[float] = None
    response_length_tokens: int
    reasoning_length_tokens: int = 0
    latency_ms: int
    finish_reason: str
    self_report: Optional[SelfReportResult] = None
    logprobs_available: bool = False
    cost_usd: Optional[float] = None
    raw_usage: dict = Field(default_factory=dict)
    generation_metadata: dict = Field(default_factory=dict)
    timestamp_utc: datetime = Field(default_factory=utc_now)


class FASComponents(BaseModel):
    logprob_score: Optional[float] = None
    enthusiasm_score: Optional[float] = None
    consistency_score: Optional[float] = None
    self_report_score: Optional[float] = None
    length_control_score: Optional[float] = None


class ProcessedScoreRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    model_slug: str
    prompt_id: str
    prompt_text: str
    category: str
    subcategory: Optional[str] = None
    difficulty: Optional[str] = None
    pair_id: Optional[str] = None
    base_task: Optional[str] = None
    framing: Optional[str] = None
    fas_score: float
    fas_components: FASComponents
    fas_weights: dict[str, float]
    reasoning_fas_score: Optional[float] = None
    reasoning_fas_components: Optional[FASComponents] = None
    reasoning_fas_weights: dict[str, float] = Field(default_factory=dict)
    reasoning_length_tokens: int = 0
    n_samples: int
    logprobs_available: bool
    computed_at: datetime = Field(default_factory=utc_now)


class CategoryFingerprint(BaseModel):
    fas_mean: float
    fas_std: float
    fas_min: float
    fas_max: float
    n_prompts: int
    reasoning_fas_mean: Optional[float] = None
    reasoning_tokens_mean: float = 0.0


class ModelFingerprint(BaseModel):
    model_slug: str
    model_display_name: str
    provider: str
    fingerprint: dict[str, CategoryFingerprint]
    overall_fas: float
    most_positive_prompt_id: Optional[str] = None
    most_positive_prompt_text: Optional[str] = None
    most_negative_prompt_id: Optional[str] = None
    most_negative_prompt_text: Optional[str] = None
    n_prompts_evaluated: int
    logprobs_coverage: float
    reasoning_coverage: float = 0.0
    reasoning_tokens_mean: float = 0.0
    run_date: str
    run_id: str
