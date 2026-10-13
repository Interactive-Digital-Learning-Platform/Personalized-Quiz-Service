import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 5.0


async def fetch_clerk_username(clerk_id: str) -> str | None:
    """The user's username straight from Clerk's backend API, or None when no
    CLERK_SECRET_KEY is configured or Clerk can't be reached -- callers then
    keep whatever username is already stored rather than guessing."""
    if not settings.CLERK_SECRET_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                f"{settings.CLERK_API_URL}/users/{clerk_id}",
                headers={"Authorization": f"Bearer {settings.CLERK_SECRET_KEY}"},
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Clerk user lookup failed for %s: %s", clerk_id, exc)
        return None
    return resp.json().get("username") or None
