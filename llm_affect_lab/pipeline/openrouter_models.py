"""OpenRouter model metadata helpers."""

from __future__ import annotations

from functools import lru_cache
import json
import urllib.request


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@lru_cache(maxsize=1)
def fetch_openrouter_models() -> dict[str, dict]:
    with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as response:
        payload = json.load(response)
    return {model["id"]: model for model in payload.get("data", [])}


def supported_parameters(model_slug: str) -> set[str]:
    model = fetch_openrouter_models().get(model_slug)
    if not model:
        return set()
    return set(model.get("supported_parameters") or [])


def supports_logprobs(model_slug: str) -> bool:
    params = supported_parameters(model_slug)
    return "logprobs" in params and "top_logprobs" in params


def supports_reasoning(model_slug: str) -> bool:
    params = supported_parameters(model_slug)
    return "reasoning" in params or "include_reasoning" in params

