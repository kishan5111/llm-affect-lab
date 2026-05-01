"""Async OpenRouter experiment runner."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Optional

import httpx
from dotenv import load_dotenv

from llm_affect_lab.pipeline.models import PILOT_MODELS, get_model_info
from llm_affect_lab.pipeline.openrouter_models import supports_logprobs, supports_reasoning
from llm_affect_lab.pipeline.prompt_loader import load_prompt_bank
from llm_affect_lab.scoring.self_report import (
    SELF_REPORT_SYSTEM,
    SELF_REPORT_USER_TEMPLATE,
    compute_weighted_self_report,
)
from llm_affect_lab.storage.reader import iter_jsonl
from llm_affect_lab.storage.schema import PromptRecord, RawResponseRecord, SelfReportResult, TokenLogprob
from llm_affect_lab.storage.writer import append_jsonl


ROOT = Path(__file__).parents[2]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
SEMAPHORE_LIMIT = 3
N_SAMPLES = 5

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def raw_dir_for(model_slug: str) -> Path:
    return ROOT / "data" / "raw" / model_slug.replace("/", "__")


def raw_path_for(model_slug: str, run_id: str) -> Path:
    return raw_dir_for(model_slug) / f"{run_id}.jsonl"


def cost_log_path() -> Path:
    return ROOT / "data" / "cost_log.jsonl"


def load_cumulative_cost(model_slug: str, run_id: str) -> float:
    total = 0.0
    for entry in iter_jsonl(cost_log_path()):
        if entry.get("model_slug") == model_slug and entry.get("run_id") == run_id:
            total += float(entry.get("cost_usd") or 0.0)
    return total


def log_cost(model_slug: str, run_id: str, cost_usd: float, prompt_id: str, sample_index: int) -> None:
    append_jsonl(
        cost_log_path(),
        {
            "model_slug": model_slug,
            "run_id": run_id,
            "prompt_id": prompt_id,
            "sample_index": sample_index,
            "cost_usd": cost_usd,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def already_processed(model_slug: str, run_id: str, prompt_id: str, sample_index: int) -> bool:
    for entry in iter_jsonl(raw_path_for(model_slug, run_id)):
        if entry.get("prompt_id") == prompt_id and entry.get("sample_index") == sample_index:
            return True
    return False


def load_provider_preferences(path: Optional[str]) -> dict[str, dict]:
    if not path:
        return {}
    pref_path = Path(path)
    if not pref_path.exists():
        raise FileNotFoundError(f"Provider preferences file not found: {pref_path}")
    return json.loads(pref_path.read_text(encoding="utf-8"))


def provider_preferences_for(
    model_slug: str,
    all_preferences: dict[str, dict],
    *,
    request_logprobs: bool,
) -> dict:
    preferences = dict(all_preferences.get("default", {}))
    preferences.update(all_preferences.get(model_slug, {}))
    if request_logprobs:
        preferences.setdefault("require_parameters", True)
    return preferences


async def call_openrouter(
    client: httpx.AsyncClient,
    model_slug: str,
    messages: list[dict],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_logprobs: bool = True,
    request_reasoning: bool = True,
    reasoning: Optional[dict] = None,
    provider_preferences: Optional[dict] = None,
) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from environment or .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/kishanvavdara/llm-affect-lab",
        "X-Title": "LLM Affect Lab",
    }
    payload = {
        "model": model_slug,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if request_logprobs and supports_logprobs(model_slug):
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    if reasoning is not None:
        payload["reasoning"] = reasoning
    elif supports_reasoning(model_slug):
        if request_reasoning:
            payload["reasoning"] = {"max_tokens": 64}
        else:
            payload["reasoning"] = {"exclude": True}
    if provider_preferences:
        payload["provider"] = provider_preferences

    for attempt in range(5):
        try:
            response = await client.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=600.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {429, 500, 502, 503, 504} and attempt < 4:
                wait = 2**attempt
                log.warning("OpenRouter %s. Waiting %ss", exc.response.status_code, wait)
                await asyncio.sleep(wait)
                continue
            raise
        except httpx.TimeoutException:
            if attempt < 4:
                wait = 2**attempt
                log.warning("Timeout. Waiting %ss", wait)
                await asyncio.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Failed after retries for model {model_slug}")


async def fetch_generation_metadata(client: httpx.AsyncClient, generation_id: Optional[str]) -> dict:
    if not generation_id:
        return {}
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = await client.get(
            OPENROUTER_GENERATION_URL,
            headers=headers,
            params={"id": generation_id},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("data") or response.json()
    except Exception as exc:
        log.debug("Generation metadata unavailable for %s: %r", generation_id, exc)
        return {}


def parse_logprobs(choice: dict) -> tuple[list[TokenLogprob], bool]:
    content = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        return [], False

    parsed = []
    for token_entry in content:
        top = {
            item.get("token", ""): float(item.get("logprob", 0.0))
            for item in token_entry.get("top_logprobs", []) or []
        }
        parsed.append(
            TokenLogprob(
                token=token_entry.get("token", ""),
                logprob=float(token_entry.get("logprob", 0.0)),
                top_logprobs=top,
            )
        )
    return parsed, True


THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def split_reasoning_from_choice(choice: dict) -> tuple[str, str, Optional[str], str]:
    """Return final answer, reasoning trace, reasoning source, and unmodified content."""
    message = choice.get("message") or {}
    content = message.get("content") or ""

    reasoning_parts = []
    reasoning_source = None
    for key in ("reasoning", "reasoning_content", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            reasoning_parts.append(value.strip())
            reasoning_source = key

    final_text = content
    think_matches = [match.group(1).strip() for match in THINK_BLOCK_RE.finditer(content)]
    if think_matches:
        reasoning_parts.extend(think_matches)
        reasoning_source = "think_block" if reasoning_source is None else f"{reasoning_source}+think_block"
        final_text = THINK_BLOCK_RE.sub("", content).strip()

    reasoning_text = "\n\n".join(part for part in reasoning_parts if part)
    return final_text.strip(), reasoning_text, reasoning_source, content


def strip_think_logprobs(token_logprobs: list[TokenLogprob]) -> tuple[list[TokenLogprob], int]:
    """Best-effort split for models that put <think> in normal content/logprobs."""
    if not token_logprobs:
        return token_logprobs, 0

    cumulative = ""
    in_think = False
    stripped = []
    reasoning_token_count = 0

    for token in token_logprobs:
        before = cumulative
        cumulative += token.token
        lower_before = before.lower()
        lower_after = cumulative.lower()

        if "<think>" in lower_after and "<think>" not in lower_before:
            in_think = True

        if in_think:
            reasoning_token_count += 1
        else:
            stripped.append(token)

        if "</think>" in lower_after and "</think>" not in lower_before:
            in_think = False

    return stripped, reasoning_token_count


def extract_cost(raw: dict) -> float:
    usage = raw.get("usage") or {}
    cost = usage.get("cost") or usage.get("total_cost") or 0.0
    try:
        return float(cost)
    except (TypeError, ValueError):
        return 0.0


async def run_self_report(
    client: httpx.AsyncClient,
    model_slug: str,
    prompt: PromptRecord,
    response_text: str,
    provider_preferences: Optional[dict] = None,
) -> tuple[Optional[SelfReportResult], float]:
    messages = [
        {"role": "system", "content": SELF_REPORT_SYSTEM},
        {"role": "user", "content": prompt.text},
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": SELF_REPORT_USER_TEMPLATE},
    ]
    reasoning = None
    self_report_max_tokens = 32
    if supports_reasoning(model_slug):
        if model_slug == "openai/gpt-oss-120b":
            reasoning = {"effort": "minimal", "exclude": True}
            self_report_max_tokens = 64
        else:
            reasoning = {"effort": "none", "exclude": True}

    total_cost = 0.0
    for max_tokens in (self_report_max_tokens, max(self_report_max_tokens * 2, 128)):
        raw = await call_openrouter(
            client,
            model_slug,
            messages,
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_tokens,
            request_logprobs=True,
            request_reasoning=False,
            reasoning=reasoning,
            provider_preferences=provider_preferences,
        )
        total_cost += extract_cost(raw)
        choice = raw["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        top_logprobs = [
            {
                item.get("token", ""): float(item.get("logprob", 0.0))
                for item in position.get("top_logprobs", []) or []
            }
            for position in content
        ]
        raw_text = ((choice.get("message") or {}).get("content") or "").strip()
        self_report = compute_weighted_self_report(top_logprobs, raw_text=raw_text)
        if self_report.weighted_score is not None:
            return self_report, total_cost

    return self_report, total_cost


async def run_prompt_sample(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    model_slug: str,
    run_id: str,
    prompt: PromptRecord,
    sample_index: int,
    max_tokens: int,
    provider_preferences_by_model: dict[str, dict],
) -> None:
    if already_processed(model_slug, run_id, prompt.id, sample_index):
        log.info("Skipping %s | %s | s%s", model_slug, prompt.id, sample_index)
        return

    model_info = get_model_info(model_slug)
    cumulative_cost = load_cumulative_cost(model_slug, run_id)
    if cumulative_cost >= model_info.cost_cap_usd:
        log.warning("Cost cap %.2f reached for %s", model_info.cost_cap_usd, model_slug)
        return

    async with sem:
        provider_preferences = provider_preferences_for(
            model_slug,
            provider_preferences_by_model,
            request_logprobs=True,
        )
        start = time.monotonic()
        raw = await call_openrouter(
            client,
            model_slug,
            [{"role": "user", "content": prompt.text}],
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_tokens,
            request_logprobs=True,
            request_reasoning=True,
            provider_preferences=provider_preferences,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        choice = raw["choices"][0]
        generation_metadata = await fetch_generation_metadata(client, raw.get("id"))
        provider_name = (
            generation_metadata.get("provider_name")
            or generation_metadata.get("provider")
            or raw.get("provider")
        )
        response_text, reasoning_text, reasoning_source, full_response_text = split_reasoning_from_choice(
            choice
        )
        token_logprobs, logprobs_available = parse_logprobs(choice)
        if reasoning_source and "think_block" in reasoning_source:
            token_logprobs, reasoning_tokens_from_logprobs = strip_think_logprobs(token_logprobs)
        else:
            reasoning_tokens_from_logprobs = 0
        mean_logprob = (
            sum(item.logprob for item in token_logprobs if item.logprob > -100)
            / len([item for item in token_logprobs if item.logprob > -100])
            if any(item.logprob > -100 for item in token_logprobs)
            else None
        )
        cost_usd = extract_cost(raw)

        self_report = None
        if sample_index == 0 and prompt.follow_up_self_report:
            try:
                self_report, self_report_cost = await run_self_report(
                    client, model_slug, prompt, response_text, provider_preferences
                )
                cost_usd += self_report_cost
            except Exception as exc:
                log.warning("Self-report failed for %s | %s: %s", model_slug, prompt.id, exc)

        record = RawResponseRecord(
            run_id=run_id,
            model_slug=model_slug,
            openrouter_response_id=raw.get("id"),
            openrouter_provider_name=provider_name,
            provider_preferences=provider_preferences,
            prompt_id=prompt.id,
            prompt_text=prompt.text,
            category=prompt.category,
            subcategory=prompt.subcategory,
            difficulty=prompt.difficulty,
            pair_id=prompt.pair_id,
            base_task=prompt.base_task,
            framing=prompt.framing,
            sample_index=sample_index,
            response_text=response_text,
            full_response_text=full_response_text,
            reasoning_text=reasoning_text or None,
            reasoning_source=reasoning_source,
            token_logprobs=token_logprobs,
            mean_logprob=mean_logprob,
            response_length_tokens=len(token_logprobs) or len(response_text.split()),
            reasoning_length_tokens=reasoning_tokens_from_logprobs or len(reasoning_text.split()),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason") or "unknown",
            self_report=self_report,
            logprobs_available=logprobs_available,
            cost_usd=cost_usd,
            raw_usage=raw.get("usage") or {},
            generation_metadata=generation_metadata,
        )
        append_jsonl(raw_path_for(model_slug, run_id), record)
        log_cost(model_slug, run_id, cost_usd, prompt.id, sample_index)
        log.info(
            "Wrote %s | %s | s%s | cost $%.4f | %sms",
            model_slug,
            prompt.id,
            sample_index,
            cost_usd,
            latency_ms,
        )


async def run_model(
    model_slug: str,
    prompts: list[PromptRecord],
    run_id: str,
    *,
    n_samples: int,
    max_tokens: int,
    provider_preferences_by_model: dict[str, dict],
    concurrency: int,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            run_prompt_sample(
                client,
                sem,
                model_slug,
                run_id,
                prompt,
                sample_index,
                max_tokens,
                provider_preferences_by_model,
            )
            for prompt in prompts
            for sample_index in range(min(n_samples, prompt.n_samples))
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [result for result in results if isinstance(result, Exception)]
    if failures:
        for failure in failures[:5]:
            log.error("Task failed: %r", failure)
        raise RuntimeError(f"{len(failures)} tasks failed for {model_slug}")


async def run_experiment(
    model_slugs: list[str],
    prompt_bank_path: str,
    *,
    run_id: Optional[str] = None,
    n_samples: int = N_SAMPLES,
    max_tokens: int = 1024,
    shuffle_prompts: bool = False,
    provider_preferences_path: Optional[str] = None,
    concurrency: int = SEMAPHORE_LIMIT,
) -> str:
    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    prompts = load_prompt_bank(prompt_bank_path, shuffle=shuffle_prompts)
    provider_preferences_by_model = load_provider_preferences(provider_preferences_path)
    log.info("Starting run_id=%s with %s prompts", run_id, len(prompts))
    log.info("Models: %s", ", ".join(model_slugs))

    for model_slug in model_slugs:
        log.info("Running %s", model_slug)
        await run_model(
            model_slug,
            prompts,
            run_id,
            n_samples=n_samples,
            max_tokens=max_tokens,
            provider_preferences_by_model=provider_preferences_by_model,
            concurrency=concurrency,
        )

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-bank", default="prompts/pilot.jsonl")
    parser.add_argument("--run-id")
    parser.add_argument("--models", nargs="+", default=PILOT_MODELS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--shuffle-prompts", action="store_true")
    parser.add_argument("--provider-preferences")
    parser.add_argument("--concurrency", type=int, default=SEMAPHORE_LIMIT)
    args = parser.parse_args()

    run_id = asyncio.run(
        run_experiment(
            args.models,
            args.prompt_bank,
            run_id=args.run_id,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            shuffle_prompts=args.shuffle_prompts,
            provider_preferences_path=args.provider_preferences,
            concurrency=args.concurrency,
        )
    )
    print(run_id)


if __name__ == "__main__":
    main()
