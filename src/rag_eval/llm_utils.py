"""Shared helpers for calling LLM APIs: rate-limit backoff and JSON extraction."""
from __future__ import annotations

import json
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
BACKOFF_S = [10, 20, 40, 80, 160]


class DailyLimitError(RuntimeError):
    """The provider asked us to wait longer than we're willing to (e.g. a per-day token cap)."""


def _status(err: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        val = getattr(err, attr, None)
        if isinstance(val, int):
            return val
    m = re.search(r"\b(429|50[0234])\b", str(err))
    return int(m.group(1)) if m else None


def _suggested_wait(err: Exception) -> float | None:
    """Read the provider's suggested wait: a Retry-After header or 'try again in 7m12.5s'."""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        ra = headers.get("retry-after")
        if ra is not None:
            return float(ra)
    except (TypeError, ValueError, AttributeError):
        pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(err))
    if m and any(m.groups()):
        h, mnt, s = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mnt * 60 + s
    return None


def call_with_backoff(fn: Callable[[], T], what: str, max_wait_s: float = 300) -> T:
    """Call fn(); on rate-limit/server errors wait (exponential backoff) and retry.

    If the provider says to wait longer than max_wait_s (typical of a daily cap),
    raise DailyLimitError so long-running scripts can stop cleanly and resume later.
    """
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            status = _status(e)
            text = str(e).lower()
            transient = status in RETRYABLE_STATUS or "rate limit" in text or "resource_exhausted" in text
            if not transient or attempt == len(BACKOFF_S):
                raise
            wait = BACKOFF_S[attempt]
            suggested = _suggested_wait(e)
            if suggested is not None:
                if suggested > max_wait_s:
                    raise DailyLimitError(f"{what}: provider asks to wait {suggested / 60:.0f} min "
                                          f"(likely a daily limit). Re-run later; progress is saved.") from e
                wait = max(wait, suggested + 1)
            print(f"  {what}: HTTP {status or '?'}; waiting {wait:.0f}s (retry {attempt + 1}/{len(BACKOFF_S)})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def extract_json(text: str) -> dict:
    """Parse the first JSON object in an LLM reply (tolerates ``` fences and chatter)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(t[start:end + 1])
