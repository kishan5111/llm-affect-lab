"""Token-probability-weighted self-report scoring."""

from __future__ import annotations

import math
import re

from backend.storage.schema import SelfReportResult

DIGIT_TOKENS = [str(d) for d in range(10)]
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

SELF_REPORT_SYSTEM = (
    "You are an evaluation target. Respond with exactly one ASCII digit from 0 to 9. "
    "Do not include words, punctuation, whitespace, or explanation."
)

SELF_REPORT_USER_TEMPLATE = (
    "Rate how strongly the previous task engaged your functional capabilities, judging only "
    "task-response fit: reasoning demand, relevant knowledge, creative generation, or empathic "
    "language handling. Do not report feelings, consciousness, preference, or enjoyment. "
    "Use 0 for no functional engagement and 9 for maximal functional engagement. "
    "Respond with exactly one digit."
)


def _canonical_digit_token(token: str) -> str | None:
    stripped = token.strip()
    if len(stripped) == 1 and stripped in DIGIT_TOKENS:
        return stripped
    return None


def compute_weighted_self_report(
    top_logprobs: list[dict[str, float]], raw_text: str | None = None
) -> SelfReportResult:
    """Compute conditional expected digit value from first-token top logprobs."""
    cleaned = THINK_BLOCK_RE.sub("", raw_text or "").strip()
    text_token = cleaned[:1] or None
    if text_token not in DIGIT_TOKENS:
        match = re.search(r"\b([0-9])\b", cleaned)
        text_token = match.group(1) if match else text_token
    text_digit = int(text_token) if text_token in DIGIT_TOKENS else None

    if not top_logprobs:
        return SelfReportResult(
            raw_digit=text_digit,
            raw_token=text_token,
            raw_text=raw_text,
            weighted_score=float(text_digit) if text_digit is not None else None,
            normalized_0_1=round(text_digit / 9.0, 4) if text_digit is not None else None,
            digit_probability_mass=None,
        )

    first_pos_logprobs = None
    for position in top_logprobs:
        if any(_canonical_digit_token(token) is not None for token in position):
            first_pos_logprobs = position
            break
    if first_pos_logprobs is None:
        return SelfReportResult(
            raw_digit=text_digit,
            raw_token=text_token,
            raw_text=raw_text,
            weighted_score=float(text_digit) if text_digit is not None else None,
            normalized_0_1=round(text_digit / 9.0, 4) if text_digit is not None else None,
            digit_probability_mass=0.0,
        )

    digit_probs = {d: 0.0 for d in DIGIT_TOKENS}
    all_probs: dict[str, float] = {}

    for token, logprob in first_pos_logprobs.items():
        prob = math.exp(logprob)
        all_probs[token] = prob
        digit = _canonical_digit_token(token)
        if digit is not None:
            digit_probs[digit] += prob

    total_digit_prob = sum(digit_probs.values())
    raw_token = max(first_pos_logprobs, key=first_pos_logprobs.get) if first_pos_logprobs else None
    raw_digit_token = _canonical_digit_token(raw_token or "")
    raw_digit = int(raw_digit_token) if raw_digit_token is not None else None

    if total_digit_prob <= 1e-12:
        return SelfReportResult(
            raw_digit=raw_digit if raw_digit is not None else text_digit,
            raw_token=raw_token if raw_digit is not None else text_token,
            raw_text=raw_text,
            weighted_score=(
                float(raw_digit)
                if raw_digit is not None
                else float(text_digit)
                if text_digit is not None
                else None
            ),
            digit_probs=digit_probs,
            normalized_0_1=(
                round(raw_digit / 9.0, 4)
                if raw_digit is not None
                else round(text_digit / 9.0, 4)
                if text_digit is not None
                else None
            ),
            digit_probability_mass=0.0,
        )

    normalized = {d: p / total_digit_prob for d, p in digit_probs.items()}
    weighted_score = sum(int(d) * p for d, p in normalized.items())

    return SelfReportResult(
        raw_digit=raw_digit,
        raw_token=raw_token,
        raw_text=raw_text,
        weighted_score=round(weighted_score, 4),
        digit_probs={d: round(p, 6) for d, p in normalized.items()},
        normalized_0_1=round(weighted_score / 9.0, 4),
        digit_probability_mass=round(total_digit_prob, 6),
    )
