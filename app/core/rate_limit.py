"""Per-user sliding-window rate limiting for the endpoints that call Groq on
every request (/quiz/generate, /analytics/feedback, /quiz/motivational-quote).

In-memory and per process: enough to stop one user from burning through the
Groq quota from a single app instance, not a global guarantee across several
workers/replicas -- that would need a shared store such as Redis.
"""
import math
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Depends, HTTPException, status

from app.core.config import settings
from app.core.security import get_current_user

_hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def _check(bucket: str, user_key: str, limit: int, window: float, now: float) -> None:
    hits = _hits[(bucket, user_key)]
    while hits and now - hits[0] >= window:
        hits.popleft()
    if len(hits) >= limit:
        retry_after = max(1, math.ceil(window - (now - hits[0]))) if hits else math.ceil(window)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests — please wait a moment and try again.",
            headers={"Retry-After": str(retry_after)},
        )
    hits.append(now)


def rate_limit(bucket: str, limit_setting: str) -> Callable:
    """Dependency enforcing `settings.<limit_setting>` requests per
    RATE_LIMIT_WINDOW_SECONDS per user for `bucket`. The limit is read at
    request time so tests (and live config reloads) can change it."""

    async def dependency(current_user: dict = Depends(get_current_user)) -> None:
        _check(
            bucket,
            current_user.get("sub", ""),
            getattr(settings, limit_setting),
            settings.RATE_LIMIT_WINDOW_SECONDS,
            time.monotonic(),
        )

    return dependency


def reset() -> None:
    """Clear all counters (tests)."""
    _hits.clear()
