"""
services/analytics/timing.py
────────────────────────────────
Lightweight phase-timing instrumentation for GET /analytics/me, active only
in development — see orchestrator.py, which wraps its query phase and each
of the 8 service phases in this context manager. Never adds overhead in
production (the timer isn't even started when disabled), and never raises —
a timing bug must not be able to break the actual analytics response.
"""
import logging
import time
from contextlib import contextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)


@contextmanager
def timed_phase(name: str):
    if settings.ENVIRONMENT != "development":
        yield
        return

    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info("GET /analytics/me — %s took %.2fms", name, elapsed_ms)
