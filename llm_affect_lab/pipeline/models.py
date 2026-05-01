"""Model registry for the API-study phases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    slug: str
    display_name: str
    provider: str
    tier: str
    cost_cap_usd: float = 5.0
    notes: str = ""


PILOT_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "qwen/qwen-2.5-72b-instruct",
]


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "meta-llama/llama-3.3-70b-instruct": ModelInfo(
        slug="meta-llama/llama-3.3-70b-instruct",
        display_name="Llama 3.3 70B Instruct",
        provider="Meta",
        tier="pilot",
    ),
    "google/gemma-3-27b-it": ModelInfo(
        slug="google/gemma-3-27b-it",
        display_name="Gemma 3 27B IT",
        provider="Google",
        tier="pilot",
    ),
    "qwen/qwen-2.5-72b-instruct": ModelInfo(
        slug="qwen/qwen-2.5-72b-instruct",
        display_name="Qwen 2.5 72B Instruct",
        provider="Qwen",
        tier="pilot",
    ),
    "qwen/qwen3-6b-flash": ModelInfo(
        slug="qwen/qwen3-6b-flash",
        display_name="Qwen3 6B Flash",
        provider="Qwen",
        tier="fast",
    ),
    "openai/gpt-4o": ModelInfo(
        slug="openai/gpt-4o",
        display_name="GPT-4o",
        provider="OpenAI",
        tier="flagship",
        cost_cap_usd=10.0,
    ),
    "anthropic/claude-opus-4": ModelInfo(
        slug="anthropic/claude-opus-4",
        display_name="Claude Opus 4",
        provider="Anthropic",
        tier="flagship",
        cost_cap_usd=15.0,
    ),
}


def get_model_info(slug: str) -> ModelInfo:
    return MODEL_REGISTRY.get(
        slug,
        ModelInfo(slug=slug, display_name=slug, provider="Unknown", tier="unregistered"),
    )
