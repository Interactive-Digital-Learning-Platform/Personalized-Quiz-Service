import logging
import time
from contextlib import contextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)


@contextmanager
def timed_phase(name: str):
    # Logs how long a phase of GET /analytics/me took, dev-only — the timer
    # doesn't even start in production, and this must never raise, since a
    # timing bug should never be able to break the actual response.
    if settings.ENVIRONMENT != "development":
        yield
        return

    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info("GET /analytics/me — %s took %.2fms", name, elapsed_ms)
